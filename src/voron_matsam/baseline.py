"""Auditable Cellpose-SAM baseline CLI. No Voronoi changes in this runner."""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .baseline_data import (FRACTIONS, MODEL_SHA256, compact_labels, disk, download_data, download_weights,
                            load_manifest, prepare_dataset, sha256, write_json)
from .jeffries import evaluate_pair, mape


def augmentations():
    import albumentations as A
    # Pin 1.3.1: v2 changed noise/dropout parameter semantics.
    return A.Compose([
        A.HorizontalFlip(p=.5), A.VerticalFlip(p=.5), A.RandomRotate90(p=.5),
        A.ElasticTransform(alpha=60, sigma=8, p=.4),
        A.GridDistortion(num_steps=5, distort_limit=.2, p=.3),
        A.RandomBrightnessContrast(brightness_limit=.3, contrast_limit=.3, p=.8),
        A.RandomGamma(gamma_limit=(70, 130), p=.5),
        A.CLAHE(clip_limit=3, tile_grid_size=(8, 8), p=.4),
        A.GaussianBlur(blur_limit=(3, 7), p=.4),
        A.GaussNoise(var_limit=(10., 60.), p=.5),
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, fill_value=0, p=.3),
    ])


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment():
    import torch
    packages = {}
    for name in ("cellpose", "torch", "torchvision", "albumentations", "numpy", "scipy",
                 "tifffile", "scikit-image", "opencv-python-headless", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                slurm={key: os.environ.get(key) for key in
                       ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION")})


def check_runtime(cpu=False):
    import torch
    for name, expected in (("cellpose", "4.0.6"), ("albumentations", "1.3.1")):
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Run inside a Unity GPU allocation; --cpu is debugging only")


def load_training_data(ids, records, root, multiplier):
    import tifffile
    transform = augmentations()
    images, labels = [], []
    for sid in ids:
        record = records[sid]
        with Image.open(root / record["image"]) as file:
            image = np.array(file.convert("RGB"))
        mask = tifffile.imread(root / record["mask"])
        for i in range(multiplier):
            if i:
                augmented = transform(image=image, mask=mask.astype(np.int32))
                img, lbl = augmented["image"], compact_labels(augmented["mask"])
            else:
                img, lbl = image.copy(), mask.copy()
            if np.unique(lbl[lbl > 0]).size < 5:
                raise ValueError(f"Augmentation {i} of {sid} has fewer than 5 masks")
            images.append(img)
            labels.append(lbl)
    return images, labels


def fit_model(model, images, labels, output, epochs, batch_size):
    from cellpose import train
    # Released baseline optimizes the entire network, not just extra heads.
    # The fixed test set is never passed to training or checkpoint selection.
    return train.train_seg(
        model.net, train_data=images, train_labels=labels,
        learning_rate=1e-6, weight_decay=.2, n_epochs=epochs, batch_size=batch_size,
        channel_axis=-1, normalize=True, rescale=False, bsize=256,
        min_train_masks=5, save_path=str(output), save_every=10,
        model_name="cellpose_grain_316L")


def score_segmentation(truth, prediction):
    from .paper_metrics import score_instances
    return score_instances(truth, prediction)


def aggregate_jeffries(rows):
    summaries = []
    for mode in ("superimposed", "independent"):
        for target in range(10, 101, 10):
            subset = [row for row in rows if row["mode"] == mode and row["target"] == target]
            if not subset:
                continue
            summary = dict(mode=mode, target=target, images=len(subset),
                           gt_target_met=sum(r["gt"]["target_met"] for r in subset),
                           pred_target_met=sum(r["pred"]["target_met"] for r in subset))
            for metric in ("internal", "intercepted", "na", "g"):
                gt = [r["gt"][metric] for r in subset]
                pred = [r["pred"][metric] for r in subset]
                for name, values in (("gt", gt), ("pred", pred)):
                    summary[f"{metric}_{name}_mean"] = (float(np.mean(values))
                                                         if all(v is not None for v in values) else None)
                summary[f"{metric}_mape"] = mape(gt, pred)
            summaries.append(summary)
    return summaries


def run_baseline(args):
    import tifffile
    from cellpose import io, models

    manifest, records, root = load_manifest(args.manifest)
    check_runtime(args.cpu)
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Pre-download cpsam with hpc/fetch_assets.sh; implicit downloads are disabled")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("Initial cpsam checkpoint differs from the audited model checksum")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    io.logger_setup()
    metadata = dict(status="started", arguments=vars(args), environment=environment(),
                    manifest_sha256=sha256(args.manifest), checkpoint_sha256=sha256(checkpoint),
                    source_sha256={file.name: sha256(file) for file in sorted(Path(__file__).parent.glob('*.py'))},
                    protocol="reconstructed_cellpose_baseline", trainable="full_network",
                    manifest_split_status=manifest["split_status"])
    metadata['label_policy'] = manifest.get('label_policy', 'released_label_before_erosion')
    metadata['evaluator'] = 'paper_metrics_v1_outer_disk2_tolerance3'
    write_json(output / "run.json", metadata)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    model = models.CellposeModel(gpu=not args.cpu, pretrained_model=str(checkpoint),
                                 use_bfloat16=not args.float32)
    metadata['model_parameter_dtype'] = str(getattr(model.net, 'dtype', 'unavailable'))
    if args.split != "zero":
        images, labels = load_training_data(manifest["fractions"][args.split], records,
                                             root, args.aug_multiplier)
        model_path, train_losses, test_losses = fit_model(
            model, images, labels, output, args.epochs, args.batch_size)
        np.savez(output / "losses.npz", train=train_losses, test=test_losses)
        metadata["trained_checkpoint"] = str(model_path)
        metadata["trained_checkpoint_sha256"] = sha256(model_path)
        metadata["augmented_images"] = len(images)
        del images, labels, model
        # Explicitly load the saved final model for evaluation.
        model = models.CellposeModel(gpu=not args.cpu, pretrained_model=str(model_path),
                                     use_bfloat16=not args.float32)
    for folder in ("instances", "binary_masks"):
        (output / folder).mkdir()
    if getattr(args, 'save_raw', False):
        (output / 'raw_outputs').mkdir()
    segmentation_rows, jeffries_rows = [], []
    test_ids = manifest["test"][:args.limit_test_images] if args.limit_test_images else manifest["test"]
    for sid in test_ids:
        record = records[sid]
        with Image.open(root / record["image"]) as f:
            image = np.array(f.convert("RGB"))
        truth = tifffile.imread(root / record["mask"])
        pred, flows, _ = model.eval(image, channel_axis=-1, diameter=None, batch_size=8,
                                flow_threshold=.4, cellprob_threshold=0., min_size=15,
                                max_size_fraction=.4, niter=None, augment=False,
                                tile_overlap=.1, bsize=256, normalize=True)
        if getattr(args, 'save_raw', False):
            if flows[1].shape != (2, *truth.shape) or flows[2].shape != truth.shape:
                raise ValueError(f'Unexpected Cellpose raw output shapes for {sid}')
            np.savez_compressed(output / 'raw_outputs' / f'{sid}.npz',
                                flow=flows[1].astype(np.float32),
                                cellprob=flows[2].astype(np.float32))
        pred = compact_labels(pred)
        if pred.shape != truth.shape:
            raise ValueError(f"Prediction shape mismatch for {sid}")
        scores, binary = score_segmentation(truth, pred)
        segmentation_rows.append(dict(image_id=sid, **scores))
        tifffile.imwrite(output / "instances" / f"{sid}_masks.tif", pred)
        Image.fromarray(binary.astype(np.uint8) * 255).save(output / "binary_masks" / f"{sid}_mask.png")
        if args.jeffries_gt == "raw":
            with Image.open(root / record["raw_mask"]) as f:
                raw = np.array(f.convert("L"))
            gt_binary = raw >= 128 if manifest["polarity"] == "bright" else raw < 128
        else:
            gt_binary = truth > 0
        for row in evaluate_pair(gt_binary, binary, args.pixels_per_um):
            jeffries_rows.append(dict(image_id=sid, **row))
        print(f"{sid}: AP50={scores['ap50']:.4f} mAP={scores['map_50_95']:.4f}", flush=True)
    write_json(output / "segmentation_per_image.json", segmentation_rows)
    write_json(output / "jeffries_per_image.json", jeffries_rows)
    summary = {key: float(np.mean([r[key] for r in segmentation_rows]))
               for key in ("ap50", "map_50_95", "boundary_f1", "count_error")}
    summary.update(images=len(segmentation_rows), split=args.split, seed=args.seed,
                   smoke_test_only=(args.epochs != 100 or args.aug_multiplier != 30
                                    or args.batch_size != 2 or bool(args.limit_test_images)),
                   jeffries=aggregate_jeffries(jeffries_rows))
    write_json(output / "summary.json", summary)
    metadata["status"] = "completed"
    write_json(output / "run.json", metadata)
    print(json.dumps(summary, indent=2))


def compare_results(results, targets_path, destination):
    targets = {r["split"]: r for r in csv.DictReader(open(targets_path, encoding="utf-8"))}
    comparisons = []
    for file in sorted(Path(results).glob("*/summary.json")):
        actual = json.loads(file.read_text(encoding="utf-8"))
        if actual.get("smoke_test_only"):
            continue
        expected = targets[actual["split"]]
        for metric in ("ap50", "map_50_95", "boundary_f1", "count_error"):
            comparisons.append(dict(run=file.parent.name, metric=metric, actual=actual[metric],
                                    paper=float(expected[metric]),
                                    delta=actual[metric] - float(expected[metric])))
        for entry in actual["jeffries"]:
            if entry["mode"] == "superimposed" and entry["target"] == 60:
                for metric in ("na_mape", "g_mape"):
                    measured = entry[metric]
                    comparisons.append(dict(run=file.parent.name, metric=f"{metric}_superimposed60",
                                            actual=measured, paper=float(expected[metric]),
                                            delta=measured - float(expected[metric]) if measured is not None else None))
    if not comparisons:
        raise ValueError("No completed summary.json files found")
    write_json(destination, comparisons)
    print(f"Wrote {len(comparisons)} comparisons to {destination}; no automatic success threshold assumed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download-data")
    download.add_argument("--output", default="data/raw/grains-v4.zip")
    weights = commands.add_parser("download-weights")
    weights.add_argument("--output", default="data/models/cpsam")
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--cpu", action="store_true")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--archive", default="data/raw/grains-v4.zip")
    prepare.add_argument("--output", default="data/processed/paper-bright")
    prepare.add_argument("--polarity", choices=("bright", "dark"), default="bright")
    prepare.add_argument("--split-seed", type=int, default=42)
    prepare.add_argument("--label-policy", choices=("released", "connected_uint16"), default="released")
    check = commands.add_parser("check-data")
    check.add_argument("--manifest", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--split", choices=("zero", *FRACTIONS), required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--epochs", type=int, default=100)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--aug-multiplier", type=int, default=30)
    run.add_argument("--pixels-per-um", type=float, default=2.26)
    run.add_argument("--jeffries-gt", choices=("raw", "processed"), default="processed")
    run.add_argument("--cpu", action="store_true")
    run.add_argument("--float32", action="store_true")
    run.add_argument("--save-raw", action="store_true", help="Save foreground scores and flow fields")
    run.add_argument("--limit-test-images", type=int, default=0, help="Smoke test only; excluded from paper comparisons")
    compare = commands.add_parser("compare")
    compare.add_argument("--results", default="outputs/baselines")
    compare.add_argument("--targets", default="configs/paper_targets.csv")
    compare.add_argument("--output", default="outputs/paper_comparison.json")
    args = parser.parse_args()
    if args.command == "download-data":
        download_data(args.output)
    elif args.command == "download-weights":
        download_weights(args.output)
    elif args.command == "doctor":
        check_runtime(args.cpu)
        from cellpose import models, train
        transform = augmentations()
        print(json.dumps(environment(), indent=2))
        print("Cellpose imports and augmentation construction passed")
    elif args.command == "prepare":
        prepare_dataset(args.archive, args.output, args.polarity, args.split_seed, args.label_policy)
    elif args.command == "check-data":
        manifest, _, _ = load_manifest(args.manifest)
        print(f"Verified all files; {len(manifest['records'])} images / {len(manifest['test'])} test")
    elif args.command == "run":
        if min(args.epochs, args.batch_size, args.aug_multiplier) < 1:
            parser.error("epochs, batch-size and aug-multiplier must be positive")
        if args.limit_test_images < 0:
            parser.error("limit-test-images cannot be negative")
        run_baseline(args)
    elif args.command == "compare":
        compare_results(args.results, args.targets, args.output)


if __name__ == "__main__":
    main()
