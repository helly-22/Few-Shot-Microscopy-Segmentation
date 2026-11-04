"""Audit archive reconstruction, label topology, split policy and old binary exports.

Writes evidence; never changes the legacy dataset or historical predictions.
"""

import argparse
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
import tifffile

from voron_matsam.baseline_data import (DATA_SHA256, archive_groups, load_manifest,
                                       sha256, stitch, write_json)
from voron_matsam.paper_metrics import score_instances


def topology(mask):
    ids, areas = np.unique(mask[mask > 0], return_counts=True)
    failures = {4: [], 8: []}
    compact = np.zeros(mask.shape, np.int32)
    positive = mask > 0
    compact[positive] = np.searchsorted(ids, mask[positive]) + 1
    for index, bounds in enumerate(ndi.find_objects(compact), 1):
        if bounds is None:
            continue
        piece = compact[bounds] == index
        for conn, structure in ((4, None), (8, np.ones((3, 3)))):
            _, count = ndi.label(piece, structure=structure)
            if count > 1:
                failures[conn].append(dict(id=int(ids[index - 1]), components=count))
    return dict(dtype=str(mask.dtype), instances=len(ids),
                minimum_area=int(areas.min()) if len(areas) else 0,
                fragmented_4=failures[4], fragmented_8=failures[8])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/raw/grains-v4.zip')
    parser.add_argument('--legacy', default='data/processed/paper-bright/manifest.json')
    parser.add_argument('--validated', default='data/processed/connected-v2/manifest.json')
    parser.add_argument('--diagnostic-inputs', default='outputs/two_shot_diagnostic/inputs')
    parser.add_argument('--legacy-scores', default='outputs/paper-20261103T234304Z/baselines/05_seed0/segmentation_per_image.json')
    parser.add_argument('--output', default='outputs/checklist/data_gate')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    old, old_records, old_root = load_manifest(args.legacy)
    new, records, root = load_manifest(args.validated)
    if old['train_order'] != new['train_order'] or old['test'] != new['test']:
        raise ValueError('The controlled mask experiment must keep split membership fixed')
    if set(new['validation']) & (set(new['test']) | set(new['fractions']['50'])):
        raise ValueError('Validation leaks into test or 20-shot training')
    if sha256(args.archive) != DATA_SHA256:
        raise ValueError('Archive checksum mismatch')
    audits = []
    previews = Image.new('RGB', (960, 8 * 210), 'white')
    draw = ImageDraw.Draw(previews)
    with zipfile.ZipFile(args.archive) as archive:
        groups = archive_groups(archive)
        for sid, record in records.items():
            x = np.array(Image.open(root / record['image']).convert('RGB'))
            raw = np.array(Image.open(root / record['raw_mask']).convert('L'))
            np.testing.assert_array_equal(x, stitch(archive, groups[sid]['image'], 'RGB'))
            np.testing.assert_array_equal(raw, stitch(archive, groups[sid]['mask'], 'L'))
            before = topology(tifffile.imread(old_root / old_records[sid]['mask']))
            after = topology(tifffile.imread(root / record['mask']))
            assert after['dtype'] == 'uint16' and not after['fragmented_4']
            assert after['minimum_area'] >= 200
            polarity = {}
            for name, interior in [('bright', raw >= 128), ('dark', raw < 128)]:
                eroded = ndi.binary_erosion(interior, border_value=1)
                _, counts = np.unique(ndi.label(eroded)[0], return_counts=True)
                polarity[name] = dict(pixel_fraction=float(interior.mean()),
                                      retained_components=int(np.count_nonzero(counts[1:] >= 200)))
            audits.append(dict(image_id=sid, source_pixels_verified=True,
                               legacy=before, validated=after, polarity=polarity))
            if len(audits) <= 8:
                y = (len(audits) - 1) * 210
                for col, array in enumerate((x, np.repeat(raw[..., None], 3, axis=2),
                                             np.where((raw < 128)[..., None], [255, 0, 0], x))):
                    thumb = Image.fromarray(array.astype(np.uint8))
                    thumb.thumbnail((310, 180))
                    previews.paste(thumb, (col * 320, y + 25))
                draw.text((5, y + 5), f'{sid}: original / raw mask / dark pixels in red', fill='black')
    previews.save(out / 'polarity_review.png')
    alternative = sorted(records, key=lambda sid: sid + '_stitched.png')
    random.Random(old['split_seed']).shuffle(alternative)
    split = dict(policy='retain historical IDs; change label conversion only',
                 train_order=new['train_order'], validation=new['validation'], test=new['test'],
                 fractions=new['fractions'],
                 author_filename_order_alternative=dict(train_order=alternative[:30], test=alternative[30:]),
                 author_original_membership_known=False,
                 test_previously_inspected=True,
                 generalization_limit='No pristine external test or physical specimen identifiers available',
                 thirty_shot_policy=new['validation_policy'])
    # Audit the exact original instance masks and binary images used by the old 2-shot run.
    inputs = Path(args.diagnostic_inputs)
    diagnostic_manifest = json.loads((inputs / 'manifest.json').read_text())
    dr = {r['image_id']: r for r in diagnostic_manifest['records']}
    binary_rows = []
    expected_scores = {r['image_id']: r for r in json.loads(Path(args.legacy_scores).read_text())}
    for sid in old['test']:
        candidates = list((inputs / 'predictions').glob(f'{sid}_*'))
        if len(candidates) != 1:
            raise ValueError(f'Missing/ambiguous diagnostic prediction for {sid}')
        pred = tifffile.imread(candidates[0])
        gt = tifffile.imread(inputs / 'prepared' / dr[sid]['mask'])
        scores, binary = score_instances(gt, pred)
        for key, value in scores.items():
            np.testing.assert_allclose(value, expected_scores[sid][key], atol=1e-7, rtol=0)
        binary_path = inputs / 'binary_predictions' / f'{sid}_mask.png'
        np.testing.assert_array_equal(binary, np.array(Image.open(binary_path)) > 0)
        binary_rows.append(dict(image_id=sid, **scores,
                                binary_cc4=int(ndi.label(binary)[1]),
                                binary_cc8=int(ndi.label(binary, structure=np.ones((3, 3)))[1]),
                                binary_render_sha256=sha256(binary_path)))
    write_json(out / 'per_image_data_audit.json', audits)
    write_json(out / 'split_policy.json', split)
    write_json(out / 'legacy_instance_vs_binary.json', binary_rows)
    report = dict(status='passed', archive_sha256=DATA_SHA256, matched_pairs=480,
                  micrographs=len(audits), tiles_per_micrograph=12,
                  all_reconstructed_pixels_match_source=True,
                  common_evaluator_matches_saved_2shot_scores_all_10=True,
                  original_instance_count=sum(r['pred_count'] for r in binary_rows),
                  binary_export_cc8_count=sum(r['binary_cc8'] for r in binary_rows),
                  legacy_fragmented_ids_4=sum(len(r['legacy']['fragmented_4']) for r in audits),
                  legacy_fragmented_ids_8=sum(len(r['legacy']['fragmented_8']) for r in audits),
                  validated_fragmented_ids_4=0,
                  legacy_instances=sum(r['legacy']['instances'] for r in audits),
                  validated_instances=sum(r['validated']['instances'] for r in audits),
                  validated_manifest_sha256=sha256(args.validated),
                  mask_protocol='connected_uint16; bright >=128, erode disk1, label4, min200',
                  exact_paper_reproduction=False,
                  reason='Source code/paper preprocessing differ and author split is unavailable')
    write_json(out / 'summary.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
