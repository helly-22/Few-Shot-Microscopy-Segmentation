import json
from types import ModuleType, SimpleNamespace
import sys
import numpy as np
import pytest

from voron_matsam.baseline import aggregate_jeffries, compare_results, fit_model, score_segmentation


def test_training_configuration_excludes_heldout_test(monkeypatch, tmp_path):
    called = {}
    def train_seg(net, **kwargs):
        called.update(kwargs)
        return 'weights', [], []
    cellpose = ModuleType('cellpose')
    cellpose.train = SimpleNamespace(train_seg=train_seg)
    monkeypatch.setitem(sys.modules, 'cellpose', cellpose)
    fit_model(SimpleNamespace(net=object()), ['image'], ['mask'], tmp_path, 100, 2)
    assert called['learning_rate'] == 1e-6
    assert called['weight_decay'] == .2
    assert called['n_epochs'] == 100 and called['batch_size'] == 2
    assert 'test_data' not in called and 'test_labels' not in called


def test_segmentation_contract_keeps_integer_instances_and_binary_boundaries(monkeypatch):
    cellpose = ModuleType('cellpose')
    def average_precision(gt, pred, threshold):
        assert len(threshold) == 10
        assert gt[0].max() == 2 and pred[0].max() == 2
        return np.ones((1, 10)), None, None, None
    cellpose.metrics = SimpleNamespace(average_precision=average_precision)
    monkeypatch.setitem(sys.modules, 'cellpose', cellpose)
    truth = np.ones((20, 20), dtype=int)
    truth[:, 10:] = 99
    scores, binary = score_segmentation(truth, truth)
    assert scores['ap50'] == scores['map_50_95'] == 1
    assert scores['boundary_f1'] == pytest.approx(1)
    assert scores['count_error'] == 0 and binary.dtype == bool
    assert not binary[:, 10].any()


def test_empty_measurements_are_not_silently_excluded():
    sample = dict(internal=0, intercepted=0, na=0., g=None, target_met=False)
    gt = dict(internal=60, intercepted=10, na=700., g=6.5, target_met=True)
    rows = [dict(mode='superimposed', target=60, gt=gt, pred=sample)]
    summary = aggregate_jeffries(rows)[0]
    assert summary['na_mape'] == 100
    assert summary['g_mape'] is None
    assert summary['pred_target_met'] == 0


def test_comparison_excludes_smoke_runs_and_reports_deltas(tmp_path):
    target = tmp_path / 'targets.csv'
    target.write_text('split,ap50,map_50_95,boundary_f1,count_error,na_mape,g_mape\n05,.6,.3,.7,-10,8,2\n')
    for name, smoke in [('smoke', True), ('real', False)]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / 'summary.json').write_text(json.dumps(dict(
            split='05', smoke_test_only=smoke, ap50=.65, map_50_95=.4,
            boundary_f1=.8, count_error=-5, jeffries=[])))
    out = tmp_path / 'comparison.json'
    compare_results(tmp_path, target, out)
    rows = json.loads(out.read_text())
    assert len(rows) == 4 and {r['run'] for r in rows} == {'real'}
    assert rows[0]['delta'] == pytest.approx(.05)


def test_real_augmentation_stack_preserves_label_ids(tmp_path):
    from PIL import Image
    import tifffile
    from voron_matsam.baseline import load_training_data, seed_all
    pytest.importorskip('albumentations')
    mask = np.repeat(np.arange(1, 7, dtype=np.uint16), 20)[None, :].repeat(120, axis=0)
    image = np.stack([mask.astype(np.uint8) * 30] * 3, axis=-1)
    Image.fromarray(image).save(tmp_path / 'image.png')
    tifffile.imwrite(tmp_path / 'mask.tif', mask)
    records = {'a':dict(image='image.png', mask='mask.tif')}
    seed_all(17)
    images, labels = load_training_data(['a'], records, tmp_path, 3)
    seed_all(17)
    images2, labels2 = load_training_data(['a'], records, tmp_path, 3)
    assert len(images) == len(labels) == 3
    np.testing.assert_array_equal(labels[0], mask)
    for img, lbl, img2, lbl2 in zip(images, labels, images2, labels2):
        assert img.shape[:2] == lbl.shape
        assert set(np.unique(lbl)) <= set(range(7))
        np.testing.assert_array_equal(img, img2)
        np.testing.assert_array_equal(lbl, lbl2)


def test_runner_writes_complete_outputs_and_reloads_checkpoint(monkeypatch, tmp_path):
    """Exercise orchestration with a fake GPU model; no weight downloads."""
    from PIL import Image
    import tifffile
    import voron_matsam.baseline as baseline
    from voron_matsam.baseline_data import sha256
    mask = np.zeros((60, 60), dtype=np.uint16)
    for row in range(2):
        for col in range(3):
            mask[row*30+2:row*30+27, col*20+2:col*20+17] = row*3+col+1
    records = {}
    for sid in ('train_a', 'train_b', 'test'):
        Image.fromarray(np.zeros((60, 60, 3), dtype=np.uint8)).save(tmp_path / f'{sid}.png')
        tifffile.imwrite(tmp_path / f'{sid}.tif', mask)
        records[sid] = dict(image=f'{sid}.png', mask=f'{sid}.tif', raw_mask=f'{sid}.png')
    manifest = dict(test=['test'], fractions={'05':['train_a', 'train_b']},
                    split_status='reconstructed', polarity='bright')
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text('{}')
    checkpoint = tmp_path / 'cpsam'
    checkpoint.write_bytes(b'initial checkpoint')
    loads = []
    class FakeModel:
        def __init__(self, **kwargs):
            loads.append(kwargs['pretrained_model'])
            self.net = object()
        def eval(self, image, **kwargs):
            return mask.copy(), None, None
    cellpose = ModuleType('cellpose')
    cellpose.io = SimpleNamespace(logger_setup=lambda:None)
    cellpose.models = SimpleNamespace(CellposeModel=FakeModel)
    cellpose.metrics = SimpleNamespace(average_precision=lambda *a, **k:(np.ones((1,10)),None,None,None))
    monkeypatch.setitem(sys.modules, 'cellpose', cellpose)
    monkeypatch.setattr(baseline, 'check_runtime', lambda cpu:None)
    monkeypatch.setattr(baseline, 'MODEL_SHA256', sha256(checkpoint))
    monkeypatch.setattr(baseline, 'load_manifest', lambda path:(manifest, records, tmp_path))
    monkeypatch.setattr(baseline, 'load_training_data', lambda *a:([np.zeros((60,60,3))], [mask]))
    def fit(*args):
        trained = tmp_path / 'trained_model'
        trained.write_bytes(b'trained checkpoint')
        return str(trained), np.ones(2), np.zeros(2)
    monkeypatch.setattr(baseline, 'fit_model', fit)
    output = tmp_path / 'output'
    args = SimpleNamespace(manifest=str(manifest_path), cpu=True, checkpoint=str(checkpoint),
        output=str(output), seed=0, float32=True, split='05', aug_multiplier=1,
        epochs=2, batch_size=2, limit_test_images=1, jeffries_gt='processed', pixels_per_um=2.26)
    baseline.run_baseline(args)
    assert len(loads) == 2 and loads[-1].endswith('trained_model')
    assert json.loads((output/'run.json').read_text())['status'] == 'completed'
    summary = json.loads((output/'summary.json').read_text())
    assert summary['images'] == 1 and summary['smoke_test_only']
    assert len(summary['jeffries']) == 20
    np.testing.assert_array_equal(tifffile.imread(output/'instances/test_masks.tif'), mask)
    assert (output/'losses.npz').is_file()
    with pytest.raises(FileExistsError):
        baseline.run_baseline(args)
