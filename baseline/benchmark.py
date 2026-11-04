"""Validated grain benchmark: validation-only selection, fixed ten-image test.

Separate from baseline_suite_v1. All final methods use the paper evaluator and
superimposed Jeffries target 60. No test labels enter fitting or selection.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
import tifffile

from baseline.common import distance_instances, normalize, read_pair
from voron_matsam.baseline import aggregate_jeffries, environment, seed_all
from voron_matsam.baseline_data import compact_labels, load_manifest, sha256, write_json
from voron_matsam.diagnostics import bootstrap_mean
from voron_matsam.jeffries import evaluate_pair
from voron_matsam.metrics import evaluate_instances
from voron_matsam.paper_metrics import score_instances


def connected(foreground, min_area=200):
    labels, _ = ndi.label(foreground)
    areas = np.bincount(labels.ravel())
    labels[areas[labels] < min_area] = 0
    return compact_labels(labels)


def classical(gray, config, method):
    smooth = gaussian(gray, sigma=config['sigma'], preserve_range=True)
    if np.ptp(smooth) == 0:
        return np.zeros(gray.shape, np.uint32)
    cutoff = threshold_otsu(smooth) + config['offset']
    foreground = smooth >= cutoff if config['bright'] else smooth < cutoff
    if method == 'connected_components':
        return connected(foreground, config['min_area'])
    return distance_instances(foreground, config['distance'], config['min_area'])


def classical_configs(method):
    base = dict(sigma=1., offset=0., bright=True, min_area=200, distance=12)
    values = [base, {**base, 'offset':-.05}, {**base, 'offset':.05},
              {**base, 'sigma':2.}, {**base, 'bright':False}]
    if method == 'watershed':
        values.extend([{**base, 'distance':8}, {**base, 'distance':16}])
    return values


def select_configuration(candidates, validation_ids, predict, truth):
    """Macro AP50 on validation only; deterministic first-candidate tie break."""
    if not validation_ids or not candidates:
        raise ValueError('Need validation images and candidates')
    rows = []
    for index, config in enumerate(candidates):
        scores = [score_instances(truth(sid), predict(sid, config))[0]['ap50'] for sid in validation_ids]
        rows.append(dict(index=index, config=config, ap50=float(np.mean(scores)),
                         image_ids=validation_ids, per_image_ap50=scores))
        print(f'Validation configuration {index}: AP50={np.mean(scores):.4f}', flush=True)
    best = max(rows, key=lambda row: row['ap50'])
    return best['config'], rows


def evaluate_test(predict, manifest_path, output, method, training=None):
    manifest, records, root = load_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('instances','binary_masks'):
        (output / folder).mkdir()
    rows, measurements = [], []
    for sid in manifest['test']:
        image, gt = read_pair(records[sid], root)
        raw_prediction = np.asarray(predict(image, sid))
        scores, binary = score_instances(gt, raw_prediction)
        pred = compact_labels(raw_prediction)
        extra = evaluate_instances(gt, pred)
        scores.update({key:extra[key] for key in ('pq','matched_dice','over_segmentation_rate','under_segmentation_rate')})
        denominator = int(np.count_nonzero(gt)) + int(np.count_nonzero(pred))
        scores['foreground_dice'] = (2 * np.count_nonzero((gt>0)&(pred>0)) / denominator if denominator else 1.)
        rows.append(dict(image_id=sid, **scores))
        tifffile.imwrite(output / 'instances' / f'{sid}_masks.tif', pred)
        Image.fromarray(binary.astype(np.uint8)*255).save(output / 'binary_masks' / f'{sid}_mask.png')
        measurements.extend(dict(image_id=sid, **r) for r in evaluate_pair(gt>0, binary, pixels_per_um=2.26, targets=(60,)))
        print(f'{method} {sid}: AP50={scores["ap50"]:.4f} BF1={scores["boundary_f1"]:.4f}', flush=True)
    indices = np.random.default_rng(20261104).integers(0,len(rows),(20000,len(rows)))
    means = {key:float(np.mean([row[key] for row in rows])) for key in rows[0] if key!='image_id'}
    summary = dict(protocol='grain_benchmark_v2', method=method, images=len(rows), metrics=means,
        uncertainty={key:bootstrap_mean([row[key] for row in rows],indices)
                     for key in ('ap50','map_50_95','boundary_f1','count_error','pq','foreground_dice')},
        jeffries=aggregate_jeffries(measurements), training=training,
        uncertainty_limit='Whole-image resampling conditional on one seed; no specimen or refit uncertainty')
    metadata = dict(status='completed', protocol='grain_benchmark_v2', method=method,
        manifest_sha256=sha256(manifest_path), test_ids=manifest['test'],
        validation_ids=manifest['validation'], environment=environment(),
        evaluator='paper_metrics_v1_outer_disk2_tolerance3',
        source_sha256=sha256(__file__), training=training,
        paper_reproduction_status='documented reconstruction/extension; not author-exact')
    write_json(output / 'per_image.json', rows)
    write_json(output / 'jeffries_per_image.json', measurements)
    write_json(output / 'summary.json', summary)
    write_json(output / 'run.json', metadata)
    return summary


def run_classical(args):
    manifest, records, root = load_manifest(args.manifest)
    validation = {sid:read_pair(records[sid],root) for sid in manifest['validation']}
    gray = {sid:normalize(x).mean(-1) for sid,(x,_) in validation.items()}
    config, history = select_configuration(classical_configs(args.method), manifest['validation'],
        lambda sid,c:classical(gray[sid],c,args.method), lambda sid:validation[sid][1])
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    write_json(out / 'validation_selection.json', history)
    info = dict(train_images=[], validation_images=manifest['validation'], selected_config=config,
                annotation_images_for_selection=len(validation), optimizer_updates=0)
    return evaluate_test(lambda image,sid:classical(normalize(image).mean(-1),config,args.method),
                         args.manifest,out/'test',args.method,info)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['connected_components','watershed','unet','stardist','matsam'],required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--checkpoint')
    parser.add_argument('--upstream')
    args=parser.parse_args()
    seed_all(args.seed)
    if args.method in ('connected_components','watershed'):
        run_classical(args)
    elif args.method=='unet':
        from baseline.benchmark_neural import run_unet
        run_unet(args)
    elif args.method=='stardist':
        from baseline.benchmark_neural import run_stardist
        run_stardist(args)
    else:
        from baseline.benchmark_matsam import run_matsam
        run_matsam(args)


if __name__=='__main__':
    main()
