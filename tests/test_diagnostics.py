import numpy as np
import pytest
from voron_matsam.diagnostics import overlap_topology, boundary_distance_summary, bootstrap_mean, paired_mean_comparison


def test_pure_split_and_neighbor_merge_with_arbitrary_ids():
    one = np.full((20, 20), 7, dtype=int)
    two = one.copy()
    two[:, 10:] = 31
    split = overlap_topology(one, two)
    merge = overlap_topology(two, one)
    assert split['split_gt_ids'] == [7]
    assert split['merged_pred_ids'] == []
    assert merge['split_gt_ids'] == []
    assert merge['merged_pred_ids'] == [7]
    assert merge['neighboring_pairs_in_same_prediction'] == [[7, 31, 7]]
    empty = overlap_topology(one, np.zeros_like(one))
    assert empty['majority_uncovered_gt_ids'] == [7]
    assert empty['fully_uncovered_gt_ids'] == [7]
    assert empty['nearly_uncovered_gt_ids'] == [7]
    assert empty['merged_pred_ids'] == []


def test_boundary_displacement_has_distance_signature():
    gt = np.ones((50, 50), dtype=int)
    gt[:, 22:] = 2
    pred = np.ones_like(gt)
    pred[:, 28:] = 2
    out = boundary_distance_summary(gt, pred)
    assert out['curve'][0]['recall'] == 0
    assert out['curve'][-1]['recall'] == 1
    assert out['gt_boundary_3_to_10px'] == 1
    assert out['gt_boundary_beyond_10px'] == 0


def test_paired_bootstrap_preserves_shared_image_variation():
    left = np.array([.1, .7, .4, .9])
    right = left - .002
    indices = np.random.default_rng(23).integers(0, 4, size=(2000, 4))
    paired = bootstrap_mean(left-right, indices)
    assert paired['mean'] == pytest.approx(.002)
    assert paired['ci95_low'] == pytest.approx(.002)
    assert paired['ci95_high'] == pytest.approx(.002)
    assert bootstrap_mean(left, indices)['ci95_high'] > .7


def test_paired_checks_do_not_invent_significance_from_identical_scores():
    scores = np.array([.2, .7, .3, .8])
    indices = np.random.default_rng(1).integers(0,4,size=(1000,4))
    result = paired_mean_comparison(scores,scores,indices)
    assert result['mean'] == result['paired_t_ci95_low'] == result['paired_t_ci95_high'] == 0
    assert result['exact_sign_flip_p_two_sided'] == 1
