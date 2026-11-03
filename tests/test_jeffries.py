import numpy as np
import pytest
from voron_matsam.jeffries import evaluate_pair, mape, measure, radial_extents


def test_internal_and_intercepted_counts_and_physical_scale():
    # Two connected components: one wholly inside radius 3; one crosses it.
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    mask[5, 7:10] = True
    extents = radial_extents(mask)
    result = measure(extents, target=1, pixels_per_um=2, radius=3)
    assert result['internal'] == result['intercepted'] == 1
    assert result['na'] == pytest.approx(1.5 / (np.pi * 9 / 4 / 1e6))
    scaled = measure(extents, target=1, pixels_per_um=4, radius=3)
    assert scaled['na'] == pytest.approx(4 * result['na'])
    assert scaled['g'] - result['g'] == pytest.approx(2, abs=1e-5)


def test_same_masks_agree_in_both_circle_modes():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50:4, 10:50:4] = True
    rows = evaluate_pair(mask, mask, targets=(10, 60))
    assert len(rows) == 4
    for row in rows:
        assert row['gt'] == row['pred']


def test_unreachable_target_and_empty_predictions_are_visible():
    mask = np.zeros((30, 30), dtype=bool)
    result = measure(radial_extents(mask))
    assert not result['target_met']
    assert result['na'] == 0 and result['g'] is None
    assert mape([1, 2], [0, 0]) == 100
    assert mape([0, 2], [1, 2]) is None
    assert mape([1, 2], [None, 2]) is None


def test_mape_averages_per_image_errors():
    assert mape([1, 10], [2, 10]) == 50
    assert mape([1, 10], [2, 10]) != abs(12 - 11) / 11 * 100


def test_primary_zero_shot_density_row_violates_reported_multiplier_bound():
    # This is a paper arithmetic audit, not a reproduction success assertion.
    lowest_possible_mean_density = 8.55 * (38.4 + 17.7 / 2)
    assert lowest_possible_mean_density > 400
    assert 125.7 < lowest_possible_mean_density
