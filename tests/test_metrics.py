import numpy as np
import pytest
from voron_matsam.metrics import evaluate_instances


def test_perfect_and_id_invariance():
    gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    m = evaluate_instances(gt, gt * 9)
    for key in ('pq', 'matched_dice', 'boundary_f1'):
        assert m[key] == 1
    assert m['over_segmentation_rate'] == m['under_segmentation_rate'] == 0


def test_false_split_and_false_merge_are_distinct():
    one = np.ones((8, 8), dtype=int)
    two = one.copy()
    two[:, 4:] = 2
    split = evaluate_instances(one, two)
    merge = evaluate_instances(two, one)
    assert split['over_segmentation_rate'] == 1
    assert split['under_segmentation_rate'] == 0
    assert merge['over_segmentation_rate'] == 0
    assert merge['under_segmentation_rate'] == 1
    assert split['pq'] == merge['pq'] == 0


def test_background_and_missing_objects():
    empty = np.zeros((8, 8), dtype=int)
    one = np.ones_like(empty)
    assert evaluate_instances(empty, empty)['pq'] == 1
    assert evaluate_instances(one, empty)['pq'] == 0
    assert evaluate_instances(empty, one)['pq'] == 0


def test_boundary_tolerance():
    gt = np.ones((10, 10), dtype=int)
    gt[:, 5:] = 2
    pred = np.ones_like(gt)
    pred[:, 6:] = 2
    assert evaluate_instances(gt, pred, boundary_tolerance=0)['boundary_f1'] < 1
    assert evaluate_instances(gt, pred, boundary_tolerance=1)['boundary_f1'] == 1


def test_invalid_masks():
    with pytest.raises(ValueError):
        evaluate_instances(np.zeros((3, 3)), np.zeros((3, 3)))
