import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import tifffile

from baseline.common import (METHODS, ap_metrics, classical_instances, distance_instances,
    evaluate, make_plan, normalize, prompt_points, select_run)
from baseline.run import load_config, run, summarize
from voron_matsam.baseline_data import sha256, write_json


@pytest.fixture
def manifest(tmp_path):
    labels = np.zeros((64, 64), dtype=np.uint16)
    labels[5:28, 5:28] = 1
    labels[35:58, 35:58] = 2
    image = np.repeat((labels > 0).astype(np.uint8)[..., None] * 255, 3, axis=-1)
    Image.fromarray(image).save(tmp_path / 'image.png')
    tifffile.imwrite(tmp_path / 'labels.tif', labels)
    names = [f'image{i}' for i in range(12)]
    records = []
    for sid in names:
        record = dict(image_id=sid, image='image.png', mask='labels.tif', raw_mask='image.png')
        record.update({f'sha256_{key}':sha256(tmp_path / record[key]) for key in ('image','mask','raw_mask')})
        records.append(record)
    metadata = dict(records=records, train_order=names[:10], test=names[10:],
                    fractions={'05':names[:2], '10':names[:4]}, grouping='synthetic test micrographs')
    path = tmp_path / 'manifest.json'
    write_json(path, metadata)
    return path


def test_all_requested_methods_have_configs_and_dependency_files():
    for method in METHODS:
        assert load_config(method)['method'] == method
        assert (Path('baseline/requirements') / f'{method}.txt').is_file()
    assert len(METHODS) == 7


def test_plan_is_nested_reproducible_and_leak_free(manifest, tmp_path):
    plan = make_plan(manifest, tmp_path/'plan.json', validation_count=3)
    assert plan == make_plan(manifest, tmp_path/'plan2.json', validation_count=3)
    assert len(plan['runs']) == 10
    for a, b in zip(plan['runs'][::2], plan['runs'][1::2]):
        assert set(a['train']) <= set(b['train'])
        assert not set(b['train']) & set(b['validation'] + b['test'])
        assert len(a['train']) == 2 and len(b['train']) == 5
        assert a['validation'] == b['validation']
    select_run(tmp_path/'plan.json', manifest, 2, 0)
    plan['runs'][0]['train'][0] = plan['runs'][0]['test'][0]
    write_json(tmp_path/'plan.json', plan)
    with pytest.raises(ValueError, match='leaking'):
        select_run(tmp_path/'plan.json', manifest, 2, 0)


def test_modified_manifest_invalidates_plan(manifest, tmp_path):
    make_plan(manifest, tmp_path/'plan.json', validation_count=3)
    manifest.write_text(manifest.read_text() + '\n')
    with pytest.raises(ValueError, match='modified manifest'):
        select_run(tmp_path/'plan.json', manifest, 2, 0)


def test_distance_watershed_splits_touching_shapes_and_handles_empty():
    y, x = np.ogrid[:64, :64]
    mask = ((y-32)**2 + (x-22)**2 < 14**2) | ((y-32)**2 + (x-43)**2 < 14**2)
    labels = distance_instances(mask, min_distance=12, min_area=10)
    assert labels.max() == 2
    assert labels[32,22] != labels[32,43]
    assert distance_instances(np.zeros((10,10), bool), 2, 1).max() == 0


def test_normalization_and_classical_empty_images():
    image = np.full((40, 60, 3), 17, np.uint8)
    assert np.isfinite(normalize(image)).all()
    assert np.count_nonzero(normalize(image)) == 0
    assert classical_instances(image, load_config('watershed')).max() == 0


def test_prompts_are_deterministic_image_only_xy_and_budget_limited():
    image = np.zeros((64, 64, 3), np.uint8)
    image[5:25, 10:30] = 255
    image[35:55, 35:55] = 255
    config = load_config('sam2')
    config.update(min_area=10, min_distance=8, max_prompts=1)
    a, b = prompt_points(image, config), prompt_points(image, config)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (1, 2)
    x, y = a[0].astype(int)
    assert image[y,x,0] > 0


def test_shared_metrics_handle_exact_split_merge_and_id_renaming():
    gt = np.ones((20, 20), np.uint32)
    pred = gt.copy()
    pred[:, 10:] = 99
    assert ap_metrics(gt, gt * 27)['map_50_95'] == 1
    split = evaluate(gt, pred)
    assert split['over_segmentation_rate'] == 1
    assert split['ap50'] == .5
    assert split['count_error'] == 1
    assert evaluate(pred, gt)['under_segmentation_rate'] == 1
    assert ap_metrics(gt * 0, gt * 0)['map_50_95'] == 1
    assert ap_metrics(gt, gt * 0)['map_50_95'] == 0


def test_real_classical_runner_outputs_and_summary_filtering(manifest, tmp_path):
    plan = tmp_path/'plan.json'
    make_plan(manifest, plan, validation_count=3)
    args = SimpleNamespace(method='watershed', config=None, seed=0, shots=2, smoke=False,
        prompts_dir=None, plan=str(plan), manifest=str(manifest), zero_shot=False,
        output=str(tmp_path/'result'), assets='not-needed.json', cpu=True, evaluate_split='test')
    run(args)
    metadata = json.loads((tmp_path/'result/run.json').read_text())
    summary = json.loads((tmp_path/'result/summary.json').read_text())
    assert metadata['status'] == 'completed' and metadata['annotation_images_used'] == 0
    assert summary['images'] == 2 and summary['zero_shot']
    assert len(list((tmp_path/'result/instances').glob('*.tif'))) == 2
    summarize(tmp_path, tmp_path/'combined.json')
    assert len(json.loads((tmp_path/'combined.json').read_text())) == 1
    with pytest.raises(FileExistsError):
        run(args)
    args.output, args.smoke = str(tmp_path/'smoke'), True
    run(args)
    args.output, args.smoke, args.evaluate_split = str(tmp_path/'validation'), False, 'validation'
    run(args)
    summarize(tmp_path, tmp_path/'combined.json')
    assert len(json.loads((tmp_path/'combined.json').read_text())) == 1


def test_runner_does_not_pass_test_labels_to_supervised_fit(manifest, tmp_path, monkeypatch):
    import baseline.run as runner
    plan = tmp_path/'plan.json'
    selection = make_plan(manifest, plan, validation_count=3)['runs'][0]
    seen = {}
    original = runner.read_pair
    def read(record, root):
        result = original(record, root)
        seen[id(result[1])] = record['image_id']
        return result
    class Fake:
        def fit(self, training, validation):
            assert {seen[id(y)] for _,y in training} == set(selection['train'])
            assert {seen[id(y)] for _,y in validation} == set(selection['validation'])
            return {'optimizer_updates':1}
        def predict(self, image, image_id):
            return np.zeros(image.shape[:2], np.uint32)
    monkeypatch.setattr(runner, 'read_pair', read)
    monkeypatch.setattr(runner, 'create', lambda *a:Fake())
    args = SimpleNamespace(method='unet_watershed', config=None, seed=0, shots=2, smoke=True,
        prompts_dir=None, plan=str(plan), manifest=str(manifest), zero_shot=False,
        output=str(tmp_path/'fake-unet'), assets='not-needed.json', cpu=True, evaluate_split='test')
    run(args)
