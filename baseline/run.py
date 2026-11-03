"""Run any baseline with a shared, frozen split plan and instance evaluation."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import tifffile

from baseline.common import METHODS, evaluate, make_plan, read_pair, select_run
from baseline.models import create
from voron_matsam.baseline_data import load_manifest, sha256, write_json

ROOT = Path(__file__).resolve().parent


def doctor(method, cpu=False):
    import torch
    expected = {'cellpose_sam':('cellpose', '4.0.6'), 'cellpose2':('cellpose','2.3.2'),
                'cellpose3':('cellpose','3.1.1.1'), 'stardist':('stardist','0.9.1')}
    if method in expected:
        package, version = expected[method]
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Use the {method} environment: requires {package}=={version}')
    if method == 'stardist':
        import tensorflow as tf
        from stardist.models import Config2D, StarDist2D
        if not cpu and not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('TensorFlow cannot see a GPU')
    elif method != 'watershed' and not cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available in this GPU allocation')
    if method.startswith('cellpose'):
        from cellpose import models
    elif method == 'sam2':
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    elif method == 'unet_watershed':
        from baseline.models.unet import UNet
    print(json.dumps(dict(method=method, cpu_check=cpu, environment=provenance()), indent=2))


def load_config(method, path=None):
    path = Path(path) if path else ROOT / 'configs' / f'{method}.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    if config['method'] != method:
        raise ValueError('Configuration method does not match selected baseline')
    for key in ('epochs', 'steps_per_epoch', 'batch_size', 'patch_size', 'min_area', 'min_distance', 'max_prompts'):
        if key in config and (not isinstance(config[key], int) or config[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    for key in ('foreground_threshold', 'quality_threshold', 'max_mask_fraction',
                'probability_threshold', 'nms_threshold'):
        if key in config and not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if 'overlap' in config and not 0 <= config['overlap'] < config['patch_size']:
        raise ValueError('Tile overlap must be nonnegative and smaller than patch_size')
    return config


def provenance():
    import torch
    import voron_matsam
    packages = {}
    for name in ('torch', 'torchvision', 'numpy', 'scipy', 'cellpose', 'stardist',
                 'tensorflow', 'scikit-image', 'SAM-2'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(python=sys.version, platform=platform.platform(), packages=packages,
                torch_cuda=torch.version.cuda, torch_cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                slurm_job=os.environ.get('SLURM_JOB_ID'),
                source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in ROOT.rglob('*.py')},
                core_source_sha256={p.name:sha256(p) for p in Path(voron_matsam.__file__).parent.glob('*.py')})


def run(args):
    config = load_config(args.method, args.config)
    config['seed'] = args.seed
    if args.smoke:
        config.update(epochs=2, steps_per_epoch=2, batch_size=1)
    if args.prompts_dir:
        if args.method != 'sam2':
            raise ValueError('Only SAM 2 accepts reusable prompt files')
        config['prompts_dir'] = str(Path(args.prompts_dir).resolve())
    selection = select_run(args.plan, args.manifest, args.shots, args.seed)
    manifest, records, root = load_manifest(args.manifest)
    if any(sid not in records for key in ('train', 'validation', 'test') for sid in selection[key]):
        raise ValueError('Split plan references an unknown image')
    if set(selection['test']) != set(manifest['test']):
        raise ValueError('The common test set must match the manifest holdout')
    zero_shot = args.zero_shot or args.method in ('sam2', 'watershed')
    if args.zero_shot and args.method not in ('cellpose2', 'cellpose3', 'cellpose_sam'):
        raise ValueError('--zero-shot is for pretrained Cellpose baselines')
    checkpoint = None
    if args.method in ('cellpose2', 'cellpose3', 'cellpose_sam', 'sam2'):
        registry = Path(args.assets).resolve()
        assets = json.loads(registry.read_text(encoding='utf-8'))
        entry = assets[args.method]
        checkpoint = registry.parent / entry['file']
        if sha256(checkpoint) != entry['sha256']:
            raise ValueError('Initial model checksum changed since asset preparation')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metadata = dict(status='started', method=args.method, config=config, split=selection,
                    requested_device='cpu' if args.cpu else 'gpu',
                    evaluation_split=args.evaluate_split,
                    annotation_images_used=0 if zero_shot else len(selection['train']),
                    protocol='baseline_suite_v1', manifest_sha256=sha256(args.manifest),
                    plan_sha256=sha256(args.plan), smoke_test_only=args.smoke,
                    checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                    environment=provenance())
    write_json(output / 'run.json', metadata)
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True)
    (output / 'pip-freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    adapter = create(args.method, config, output, checkpoint, args.cpu)
    if zero_shot:
        training_info = adapter.fit([], []) if args.method in ('sam2', 'watershed') else {'optimizer_updates':0}
    else:
        training = [read_pair(records[sid], root) for sid in selection['train']]
        validation = [read_pair(records[sid], root) for sid in selection['validation']]
        training_info = adapter.fit(training, validation)
    metadata['training'] = training_info
    write_json(output / 'run.json', metadata)
    (output / 'instances').mkdir()
    rows = []
    test = selection[args.evaluate_split][:1] if args.smoke else selection[args.evaluate_split]
    for sid in test:
        image, truth = read_pair(records[sid], root)
        prediction = adapter.predict(image, sid)
        scores = evaluate(truth, prediction)
        tifffile.imwrite(output / 'instances' / f'{sid}_masks.tif', prediction.astype(np.uint32))
        rows.append(dict(image_id=sid, **scores))
        print(f'{args.method} {sid}: PQ={scores["pq"]:.4f} AP50={scores["ap50"]:.4f}', flush=True)
    write_json(output / 'per_image.json', rows)
    summary = dict(method=args.method, shots=None if zero_shot else args.shots, seed=args.seed, zero_shot=zero_shot,
                   evaluation_split=args.evaluate_split,
                   images=len(rows), smoke_test_only=args.smoke, protocol='baseline_suite_v1',
                   metrics={key:float(np.mean([r[key] for r in rows])) for key in rows[0] if key != 'image_id'},
                   evaluation=dict(boundary_tolerance_pixels=1, boundary_distance='Manhattan',
                                   split_merge_overlap=.1, ap='TP/(TP+FP+FN)',
                                   averaging='macro per image; no independence assumption'))
    write_json(output / 'summary.json', summary)
    metadata['status'] = 'completed'
    write_json(output / 'run.json', metadata)


def summarize(results, destination):
    rows = []
    expected = None
    for file in sorted(Path(results).rglob('summary.json')):
        summary = json.loads(file.read_text(encoding='utf-8'))
        if (summary.get('protocol') != 'baseline_suite_v1' or summary.get('smoke_test_only')
                or summary.get('evaluation_split') != 'test'):
            continue
        metadata = json.loads((file.parent / 'run.json').read_text(encoding='utf-8'))
        if metadata['status'] != 'completed':
            continue
        signature = (metadata['manifest_sha256'], metadata['plan_sha256'])
        if expected is not None and signature != expected:
            raise ValueError('Cannot pool runs from different manifests or split plans')
        expected = signature
        rows.append(dict(run=str(file.parent), **summary))
    if not rows:
        raise ValueError('No completed, non-smoke baseline suite runs found')
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--manifest', default='data/processed/paper-bright/manifest.json')
    plan.add_argument('--output', default='outputs/baseline_plan.json')
    plan.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    plan.add_argument('--shots', type=int, nargs='+', default=[2, 5])
    plan.add_argument('--validation-count', type=int, default=5)
    job = commands.add_parser('run')
    job.add_argument('--method', choices=METHODS, required=True)
    job.add_argument('--manifest', default='data/processed/paper-bright/manifest.json')
    job.add_argument('--plan', default='outputs/baseline_plan.json')
    job.add_argument('--assets', default='data/models/baseline_assets.json')
    job.add_argument('--shots', type=int, choices=[2, 5], default=2)
    job.add_argument('--seed', type=int, default=0)
    job.add_argument('--config')
    job.add_argument('--output', required=True)
    job.add_argument('--prompts-dir')
    job.add_argument('--cpu', action='store_true')
    job.add_argument('--smoke', action='store_true')
    job.add_argument('--zero-shot', action='store_true')
    job.add_argument('--evaluate-split', choices=['validation', 'test'], default='test')
    assets = commands.add_parser('assets')
    assets.add_argument('--method', choices=METHODS, required=True)
    assets.add_argument('--output', default='data/models')
    check = commands.add_parser('doctor')
    check.add_argument('--method', choices=METHODS, required=True)
    check.add_argument('--cpu', action='store_true')
    report = commands.add_parser('summarize')
    report.add_argument('--results', default='outputs/baseline_suite')
    report.add_argument('--output', default='outputs/baseline_suite_results.json')
    args = parser.parse_args()
    if args.command == 'plan':
        make_plan(args.manifest, args.output, args.seeds, args.shots, args.validation_count)
    elif args.command == 'run':
        run(args)
    elif args.command == 'assets':
        from baseline.assets import fetch
        fetch(args.method, args.output)
    elif args.command == 'doctor':
        doctor(args.method, args.cpu)
    elif args.command == 'summarize':
        summarize(args.results, args.output)


if __name__ == '__main__':
    main()
