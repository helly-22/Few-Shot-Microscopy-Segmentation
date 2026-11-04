"""Build the post-run comparison, per-image uncertainty and 2-shot galleries."""

import argparse
import csv
import html
import importlib.util
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tifffile

from voron_matsam.baseline_data import load_manifest, sha256, write_json
from voron_matsam.diagnostics import (bootstrap_mean, boundary_distance_summary,
                                      overlap_topology, paired_mean_comparison)


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--source-root', type=Path,
        default=Path(os.environ.get('CHECKLIST_SOURCE_ROOT', Path(__file__).resolve().parents[1])))
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('checklist_plot_helpers',
        args.source_root / 'scripts/diagnose_two_shot.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    panel = helper.panel
    manifest, records, root = load_manifest(args.manifest)
    out = args.run_root / 'report'
    out.mkdir(exist_ok=False)
    plots = out / 'plots'
    plots.mkdir()
    targets = {r['split']: r for r in csv.DictReader(args.targets.open())}
    indices = np.random.default_rng(20261104).integers(0, 10, (20000, 10))
    configurations = [('zero_seed0', 'zero'), ('05_seed0', '05'), ('05_float32_seed0', '05')]
    metrics = ('ap50', 'map_50_95', 'boundary_f1', 'count_error')
    results, detailed, per_run, run_metadata = [], [], {}, {}
    manifest_hash = sha256(args.manifest)
    for name, target in configurations:
        run = args.run_root / name
        metadata, summary = load(run / 'run.json'), load(run / 'summary.json')
        run_metadata[name] = metadata
        rows = load(run / 'segmentation_per_image.json')
        if (metadata['status'] != 'completed' or metadata['manifest_sha256'] != manifest_hash
                or summary['smoke_test_only'] or [r['image_id'] for r in rows] != manifest['test']):
            raise ValueError(f'Incomplete or incompatible run: {name}')
        per_run[name] = rows
        primary = next(r for r in summary['jeffries'] if r['mode'] == 'superimposed' and r['target'] == 60)
        result = dict(run=name, published_split=target,
                      metrics={key: dict(**bootstrap_mean([r[key] for r in rows], indices),
                                        paper=float(targets[target][key])) for key in metrics},
                      primary_jeffries=primary, paper_g_mape=float(targets[target]['g_mape']),
                      model_parameter_dtype=metadata['model_parameter_dtype'],
                      environment=metadata.get('environment', {}),
                      arguments=metadata.get('arguments', {}),
                      checkpoint_sha256=metadata.get('trained_checkpoint_sha256', metadata['checkpoint_sha256']))
        jeffries = [r for r in load(run / 'jeffries_per_image.json')
                    if r['mode'] == 'superimposed' and r['target'] == 60]
        g_errors = [abs(r['pred']['g'] - r['gt']['g']) / r['gt']['g'] * 100
                    for r in jeffries if r['pred']['g'] is not None and r['gt']['g'] is not None and r['gt']['g'] > 0]
        result['g_mape_uncertainty'] = bootstrap_mean(g_errors, indices) if len(g_errors) == 10 else None
        results.append(result)
        if name == 'zero_seed0':
            continue
        for row in rows:
            sid = row['image_id']
            record = records[sid]
            image = np.array(Image.open(root / record['image']).convert('RGB'))
            gt = tifffile.imread(root / record['mask'])
            pred = tifffile.imread(run / 'instances' / f'{sid}_masks.tif')
            topology = overlap_topology(gt, pred)
            raw = np.load(run / 'raw_outputs' / f'{sid}.npz')
            probability, flow = raw['cellprob'], raw['flow']
            assert probability.shape == gt.shape and flow.shape == (2, *gt.shape)
            item = dict(run=name, **row, split_gt_count=len(topology['split_gt_ids']),
                        merged_pred_count=len(topology['merged_pred_ids']),
                        gt_foreground_below_zero=float(np.mean(probability[gt > 0] <= 0)),
                        gt_foreground_uncovered=float(np.mean(pred[gt > 0] == 0)),
                        boundary_distances=boundary_distance_summary(gt, pred))
            detailed.append(item)
            panel(image, gt, pred, topology, item, plots / f'{name}_{sid}_instances.png')
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image)
            axes[0].set_title('Original')
            heat = axes[1].imshow(probability, cmap='coolwarm', vmin=-6, vmax=6)
            axes[1].set_title('Foreground score (clipped to -6..6)')
            fig.colorbar(heat, ax=axes[1], shrink=.6)
            angle = (np.arctan2(flow[0], flow[1]) + np.pi) / (2 * np.pi)
            axes[2].imshow(angle, cmap='hsv', vmin=0, vmax=1,
                           alpha=np.clip(np.linalg.norm(flow, axis=0) / 5, 0, 1))
            axes[2].set_title('Flow direction; opacity = magnitude / 5')
            for ax in axes:
                ax.axis('off')
            fig.suptitle(f'{name}: {sid}; GT pixels below score zero {item["gt_foreground_below_zero"]:.1%}')
            fig.tight_layout()
            fig.savefig(plots / f'{name}_{sid}_raw.png', dpi=130)
            plt.close(fig)
            raw.close()
    left, right = run_metadata['05_float32_seed0'], run_metadata['05_seed0']
    for key in ('seed', 'epochs', 'aug_multiplier', 'batch_size', 'jeffries_gt', 'pixels_per_um', 'split'):
        if left.get('arguments', {}).get(key) != right.get('arguments', {}).get(key):
            raise ValueError(f'Unmatched precision-control training/evaluation setting: {key}')
    if left['checkpoint_sha256'] != right['checkpoint_sha256']:
        raise ValueError('Precision controls did not start from the same checkpoint file')
    left_gpu = left.get('environment', {}).get('gpu')
    same_gpu = bool(left_gpu and left_gpu == right.get('environment', {}).get('gpu'))
    paired = {key: paired_mean_comparison(
        [r[key] for r in per_run['05_float32_seed0']],
        [r[key] for r in per_run['05_seed0']], indices) for key in metrics}
    validations = {name: load(args.run_root / name / 'summary.json') for name in
                   ('05_validation', '05_bf16_checkpoint_fp32_validation', '05_float32_validation')}
    report = dict(results=results, per_image=detailed, float32_minus_bfloat16=paired,
                  validation=validations, manifest_sha256=manifest_hash, source_sha256=sha256(__file__),
                  precision_control_gpu_models_match=same_gpu,
                  gallery_review='automatically generated; manual review pending',
                  uncertainty='20,000 whole-image bootstrap resamples, seed 20261104; one fit per policy; '
                    'no seed/specimen variance; paired tests exploratory and unadjusted',
                  exact_paper_reproduction=False,
                  limitations=['Author split and physical calibration unconfirmed',
                               'Test set previously inspected',
                               'Canonical masks differ from released label-before-erosion preprocessing'])
    write_json(out / 'results.json', report)
    with (out / 'per_image_scores.csv').open('w', newline='') as f:
        columns = ['run', 'image_id', *metrics, 'gt_count', 'pred_count', 'split_gt_count',
                   'merged_pred_count', 'gt_foreground_below_zero', 'gt_foreground_uncovered']
        writer = csv.DictWriter(f, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(detailed)
    def estimate(value):
        return f'{value["mean"]:.4f} [{value["ci95_low"]:.4f}, {value["ci95_high"]:.4f}]'
    body = ['<!doctype html><meta charset="utf-8"><title>Checklist Cellpose results</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:15px}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px}'
        'img{max-width:100%;height:auto}details{margin:20px 0}</style>',
        '<h1>Cellpose data and precision diagnostics</h1>',
        '<p>Fixed historical ten-image test membership, connected uint16 ground truth. '
        'Published numbers are targets, not confirmed replications. Intervals describe '
        'image variability conditional on one fitted checkpoint. Manual gallery review is pending.</p>',
        '<table><tr><th>Run</th><th>AP50 [95% CI]</th><th>Paper AP50</th>'
        '<th>Boundary F1 [95% CI]</th><th>G MAPE %</th><th>Paper G MAPE %</th></tr>']
    for result in results:
        body.append(f'<tr><td>{result["run"]}</td><td>{estimate(result["metrics"]["ap50"])}</td>'
            f'<td>{result["metrics"]["ap50"]["paper"]:.4f}</td>'
            f'<td>{estimate(result["metrics"]["boundary_f1"])}</td>'
            f'<td>{result["primary_jeffries"]["g_mape"]}</td><td>{result["paper_g_mape"]}</td></tr>')
    body.append('</table><h2>Paired float32 minus bfloat16 AP50</h2><pre>'
                + html.escape(json.dumps(paired['ap50'], indent=2)) + '</pre>')
    body.append('<p>Hardware: ' + html.escape(str({r['run']: r['environment'].get('gpu') for r in results}))
                + '. Matching GPU models for precision controls: ' + str(same_gpu)
                + '. Hardware differences, if present, limit attribution to precision alone.</p>')
    body.append('<h2>Validation selections</h2><p>Five validation images; separate one-factor sweeps. '
                'Selections have not been applied to test predictions.</p><pre>' + html.escape(json.dumps(
                    {name: value['selected'] for name, value in validations.items()}, indent=2)) + '</pre>')
    body.append('<h2>Every test image, both 2-shot checkpoints</h2>')
    for row in detailed:
        prefix = f'{row["run"]}_{row["image_id"]}'
        body.append(f'<details><summary>{prefix}: AP50 {row["ap50"]:.4f}; split candidates '
            f'{row["split_gt_count"]}; merge candidates {row["merged_pred_count"]}</summary>'
            f'<img src="plots/{prefix}_instances.png"><img src="plots/{prefix}_raw.png"></details>')
    body.append('<p><a href="results.json">Full metrics, audit and uncertainty</a> | '
                '<a href="per_image_scores.csv">Per-image scores</a> | '
                '<a href="https://arxiv.org/html/2604.18957v1">Reference paper</a></p>')
    (out / 'report.html').write_text('\n'.join(body), encoding='utf-8')
    print(f'Completed report: {out / "report.html"}')


if __name__ == '__main__':
    main()
