"""Cache Cellpose validation flows and sweep one parameter at a time.

Never reads test images. The reference, explicit 0.5/99.5 percentile and
disabled normalization settings each need one network pass per image.
Threshold/min-size sweeps reuse the reference flows, with no network reruns.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile

from voron_matsam.baseline import check_runtime, environment, seed_all
from voron_matsam.baseline_data import compact_labels, load_manifest, sha256, write_json
from voron_matsam.diagnostics import bootstrap_mean
from voron_matsam.paper_metrics import score_instances


def configurations():
    default = dict(cellprob_threshold=0., flow_threshold=.4, min_size=15)
    result = [('reference', default)]
    for name, values in [('cellprob_threshold', [-3., -2., -1., 0., 1.]),
                         ('flow_threshold', [0., .4, .8, 1.2]),
                         ('min_size', [0, 15, 50, 100])]:
        for value in values:
            if value != default[name]:
                result.append((f'{name}_{value}', {**default, name: value}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--float32', action='store_true')
    args = parser.parse_args()
    check_runtime(False)
    from cellpose import dynamics, models
    manifest, records, root = load_manifest(args.manifest)
    ids = manifest['validation']
    if not ids or set(ids) & (set(manifest['test']) | set(manifest['fractions']['05'])):
        raise ValueError('Validation must be nonempty and excluded from test and 2-shot training')
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'raw_outputs').mkdir()
    (out / 'instances').mkdir()
    metadata = dict(status='started', arguments=vars(args), environment=environment(),
                    checkpoint_sha256=sha256(checkpoint), manifest_sha256=sha256(args.manifest),
                    source_sha256=sha256(__file__), partition='validation', image_ids=ids,
                    test_images_read=False, normalization_alternative_percentiles=[.5, 99.5],
                    selection='highest validation mean AP50; ties retain reference/earlier config',
                    design='one-factor-at-a-time; interactions not tested', niter=200)
    write_json(out / 'run.json', metadata)
    seed_all(args.seed)
    model = models.CellposeModel(gpu=True, pretrained_model=str(checkpoint),
                                 use_bfloat16=not args.float32)
    rows = []
    normalization_modes = [('reference', True),
                           ('percentile_0.5_99.5', dict(normalize=True, percentile=[.5, 99.5])),
                           ('none', False)]
    for sid in ids:
        record = records[sid]
        image = np.array(Image.open(root / record['image']).convert('RGB'))
        gt = tifffile.imread(root / record['mask'])
        for mode, normalization in normalization_modes:
            original, flows, _ = model.eval(image, channel_axis=-1, diameter=None,
                batch_size=8, flow_threshold=.4, cellprob_threshold=0., min_size=15,
                max_size_fraction=.4, niter=200, augment=False, tile_overlap=.1,
                bsize=256, normalize=normalization)
            flow, probability = flows[1].astype(np.float32), flows[2].astype(np.float32)
            assert flow.shape == (2, *gt.shape) and probability.shape == gt.shape
            raw_path = out / 'raw_outputs' / f'{sid}_{mode}.npz'
            np.savez_compressed(raw_path, flow=flow, cellprob=probability)
            configs = configurations() if mode == 'reference' else [('reference', configurations()[0][1])]
            for config_id, parameters in configs:
                pred = dynamics.resize_and_compute_masks(flow, probability, niter=200,
                    max_size_fraction=.4, device=model.device, **parameters)
                pred = compact_labels(pred)
                if config_id == 'reference':
                    np.testing.assert_array_equal(pred, compact_labels(original),
                        err_msg='Cached-flow reconstruction differs from native Cellpose output')
                config_key = config_id if mode == 'reference' else f'normalization_{mode}'
                scores, binary = score_instances(gt, pred)
                tifffile.imwrite(out / 'instances' / f'{sid}_{config_key}.tif', pred)
                rows.append(dict(image_id=sid, configuration=config_key, normalization=mode,
                                 parameters=parameters, **scores,
                                 gt_foreground_below_threshold=float(np.mean(
                                     probability[gt > 0] <= parameters['cellprob_threshold'])),
                                 gt_foreground_uncovered=float(np.mean(pred[gt > 0] == 0)),
                                 raw_sha256=sha256(raw_path),
                                 cellprob_quantiles=np.percentile(probability, [0, 1, 25, 50, 75, 99, 100]).tolist()))
                write_json(out / 'per_image.json', rows)
                print(f'{sid} {config_key}: AP50={scores["ap50"]:.4f}', flush=True)
    summaries = []
    resample_indices = np.random.default_rng(20261104).integers(0, len(ids), (20000, len(ids)))
    for config_key in dict.fromkeys(r['configuration'] for r in rows):
        subset = [r for r in rows if r['configuration'] == config_key]
        assert len(subset) == len(ids)
        summaries.append(dict(configuration=config_key, images=len(subset),
            parameters=subset[0]['parameters'], normalization=subset[0]['normalization'],
            **{key: float(np.mean([r[key] for r in subset]))
               for key in ('ap50', 'map_50_95', 'boundary_f1', 'count_error', 'gt_foreground_uncovered')},
            ap50_uncertainty=bootstrap_mean([r['ap50'] for r in subset], resample_indices)))
    best = max(summaries, key=lambda row: row['ap50'])
    write_json(out / 'summary.json', dict(configurations=summaries, selected=best,
        selection_partition='validation', test_application_performed=False,
        uncertainty_limit='Five images conditional on one fitted checkpoint; no seed variance'))
    metadata.update(status='completed', cached_reconstruction_checks_passed=len(ids) * 3,
                    network_image_passes=len(ids) * 3, configurations=len(summaries))
    write_json(out / 'run.json', metadata)


if __name__ == '__main__':
    main()
