import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
from scipy import ndimage as ndi
from skimage.morphology import binary_erosion

from voron_matsam.baseline_data import (archive_groups, boundary_to_instances,
    compact_labels, disk, load_manifest, sha256, stitch)


def test_mask_polarity_and_small_component_filter():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[2:25, 2:20] = 255
    mask[2:25, 25:45] = 255
    mask[30:32, 30:32] = 255
    labels = boundary_to_instances(mask)
    assert labels.max() == 2
    assert labels[10, 10] != labels[10, 30]
    assert labels[30, 30] == 0
    assert np.array_equal(labels, boundary_to_instances(255 - mask, 'dark'))


def test_optimized_erosion_matches_released_full_image_operation():
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[:25, :20] = 255
    mask[2:, 25:] = 255
    initial, n = ndi.label(mask >= 128)
    expected = np.zeros_like(initial)
    for gid in range(1, n + 1):
        eroded = binary_erosion(initial == gid, disk(1))
        if eroded.sum() >= 200:
            expected[eroded] = gid
    np.testing.assert_array_equal(boundary_to_instances(mask), compact_labels(expected))


def test_label_compaction_preserves_background_and_full_foreground():
    np.testing.assert_array_equal(compact_labels(np.array([[0, 99, 7]])), [[0, 2, 1]])
    np.testing.assert_array_equal(compact_labels(np.full((2, 2), 5)), np.ones((2, 2)))


def test_archive_requires_complete_real_grids_and_excludes_synthetic():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        for sid in range(1, 41):
            for row in range(1, 4):
                for col in range(1, 5):
                    for folder, prefix in [('RG', 'RG'), ('RGMask', 'RGMask')]:
                        z.writestr(f'GRAIN DATA SET/{folder}/{prefix}{sid}_{row}_{col}.jpg', b'')
        z.writestr('GRAIN DATA SET/AG/AG_NOISE1.png', b'')
    with zipfile.ZipFile(buffer) as z:
        groups = archive_groups(z)
    assert len(groups) == 40
    assert len(groups['RG1']['mask']) == 12
    with zipfile.ZipFile(io.BytesIO(), 'w') as z:
        with pytest.raises(ValueError, match='40 real micrographs'):
            archive_groups(z)


def test_stitch_uses_coordinate_order_and_rejects_resizing():
    buffer = io.BytesIO()
    tiles = {}
    with zipfile.ZipFile(buffer, 'w') as z:
        for row in range(1, 4):
            for col in range(1, 5):
                name = f'{row}_{col}.png'
                png = io.BytesIO()
                Image.fromarray(np.full((2, 3), row * 10 + col, dtype=np.uint8)).save(png, format='PNG')
                z.writestr(name, png.getvalue())
                tiles[row, col] = name
    with zipfile.ZipFile(buffer) as z:
        image = stitch(z, tiles, 'L')
    assert image.shape == (6, 12)
    assert image[0, 0] == 11 and image[-1, -1] == 34


def test_manifest_detects_leakage_and_changed_files(tmp_path):
    for name in ('image', 'mask', 'raw_mask'):
        (tmp_path / name).write_bytes(b'data')
    records = [dict(image_id=s, **{name: name for name in ('image', 'mask', 'raw_mask')},
                    **{f'sha256_{name}':sha256(tmp_path / name)
                       for name in ('image', 'mask', 'raw_mask')}) for s in ('a', 'b')]
    manifest = dict(records=records, train_order=['a'], test=['b'], fractions={'05':['a']})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    load_manifest(path)
    (tmp_path / 'image').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        load_manifest(path)
    manifest['test'] = ['a']
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='partition'):
        load_manifest(path, verify=False)
