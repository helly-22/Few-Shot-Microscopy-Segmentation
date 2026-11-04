import numpy as np
import pytest
from scipy import ndimage as ndi

from voron_matsam.baseline_data import connected_instances
from voron_matsam.paper_metrics import score_instances


def test_erosion_splits_bridge_into_distinct_uint16_ids():
    raw = np.zeros((40, 60), np.uint8)
    raw[3:30, 3:24] = 255
    raw[3:30, 34:55] = 255
    raw[15, 23:35] = 255
    labels = connected_instances(raw)
    assert labels.dtype == np.uint16
    assert labels.max() == 2
    for gid in (1, 2):
        assert ndi.label(labels == gid)[1] == 1
        assert np.count_nonzero(labels == gid) >= 200
    np.testing.assert_array_equal(labels, connected_instances(255 - raw, 'dark'))


def test_matching_at_exact_half_iou_and_id_invariance():
    gt = np.zeros((20, 20), np.uint16)
    gt[3:13, 3:13] = 1200
    pred = np.zeros_like(gt)
    pred[3:13, 3:8] = 60000
    scores, _ = score_instances(gt, pred)
    assert scores['ap50'] == 1
    assert scores['map_50_95'] == pytest.approx(.1)
    assert scores['gt_count'] == scores['pred_count'] == 1
    assert scores['count_error'] == 0


def test_boundary_tolerance_and_binary_export_can_alter_counts():
    gt = np.zeros((40, 40), np.uint16)
    gt[5:35, 5:20] = 1
    gt[5:35, 20:35] = 2
    scores, binary = score_instances(gt, gt)
    assert scores['boundary_f1'] == pytest.approx(1, abs=1e-6)
    assert ndi.label(gt > 0)[1] == 1  # plain foreground loses adjacent IDs
    assert ndi.label(binary)[1] == 2
    shifted = np.roll(gt, 2, axis=0)
    shifted_score, _ = score_instances(gt, shifted)
    assert shifted_score['boundary_f1'] > .99
    far_score, _ = score_instances(gt, np.roll(gt, 10, axis=0))
    assert far_score['boundary_f1'] < shifted_score['boundary_f1']


def test_rejects_invalid_instance_masks_and_defines_empty_case():
    empty = np.zeros((10, 10), np.int32)
    scores, _ = score_instances(empty, empty)
    assert scores['ap50'] == 1 and scores['count_error'] == 0
    assert scores['boundary_f1'] == 0
    for invalid in (empty.astype(float), empty - 1):
        with pytest.raises(ValueError, match='nonnegative integers'):
            score_instances(empty, invalid)


def test_validation_sweep_caches_flows_and_never_reads_test_images(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from PIL import Image
    import tifffile
    import scripts.cellpose_validation_diagnostic as diagnostic
    mask = np.zeros((40, 40), np.uint16)
    mask[5:35, 5:35] = 1
    Image.fromarray(np.zeros((40, 40, 3), np.uint8)).save(tmp_path / 'val.png')
    tifffile.imwrite(tmp_path / 'val.tif', mask)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'test checkpoint')
    manifest_file = tmp_path / 'manifest.json'
    manifest_file.write_text('{}')
    calls = dict(network=0, reconstruct=0)
    class FakeModel:
        device = 'cpu'
        def __init__(self, **kwargs):
            pass
        def eval(self, image, **kwargs):
            calls['network'] += 1
            return mask, [None, np.zeros((2, 40, 40)), np.zeros((40, 40))], None
    def reconstruct(flow, probability, **kwargs):
        calls['reconstruct'] += 1
        return mask.copy()
    monkeypatch.setitem(sys.modules, 'cellpose', SimpleNamespace(
        models=SimpleNamespace(CellposeModel=FakeModel),
        dynamics=SimpleNamespace(resize_and_compute_masks=reconstruct)))
    monkeypatch.setattr(diagnostic, 'check_runtime', lambda cpu: None)
    monkeypatch.setattr(diagnostic, 'environment', lambda: {})
    monkeypatch.setattr(diagnostic, 'load_manifest', lambda path: (
        dict(validation=['val'], test=['missing_test'], fractions={'05':['missing_train']}),
        dict(val=dict(image='val.png', mask='val.tif')), tmp_path))
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['diagnostic', '--manifest', str(manifest_file),
        '--checkpoint', str(checkpoint), '--output', str(out)])
    diagnostic.main()
    assert calls == dict(network=3, reconstruct=13)
    assert len(list((out / 'raw_outputs').glob('*.npz'))) == 3
    metadata = json.loads((out / 'run.json').read_text())
    assert metadata['status'] == 'completed' and metadata['test_images_read'] is False
    summary = json.loads((out / 'summary.json').read_text())
    assert len(summary['configurations']) == 13
    assert summary['selected']['configuration'] == 'reference'


def test_checkpoint_audit_distinguishes_dtype_rounding_from_training_updates():
    import torch
    from scripts.audit_checkpoint_updates import audit
    initial = dict(weight=torch.tensor([1.001, 2.002, 3.003], dtype=torch.float32))
    rounded = initial['weight'].to(torch.bfloat16)
    unchanged = audit(initial, dict(weight=rounded))
    assert unchanged['fraction_changed'] == 0
    changed = rounded.clone()
    changed[1] += 1
    report = audit(initial, dict(weight=changed))
    assert report['changed'] == 1
    assert report['fraction_changed'] == pytest.approx(1 / 3)
