"""Overlap topology, boundary-distance diagnostics, and paired image uncertainty.

Overlap events are candidates, not manual annotations of error mechanisms.
Image resampling describes test-image variability conditional on one fitted model.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import t as student_t
from .metrics import _labels, boundaries


def overlap_topology(truth, prediction, threshold=0.1, neighbor_distance=5):
    gt, pred = _labels(truth), _labels(prediction)
    if gt.shape != pred.shape or not gt.size:
        raise ValueError('Masks must have identical nonempty shape')
    if not 0 < threshold <= 1 or neighbor_distance < 0:
        raise ValueError('Invalid overlap threshold or neighbor distance')
    gids, gi = np.unique(gt, return_inverse=True)
    pids, pi = np.unique(pred, return_inverse=True)
    counts = np.bincount(gi.ravel()*len(pids)+pi.ravel(),
                         minlength=len(gids)*len(pids)).reshape(len(gids), len(pids))
    ga, pa = counts.sum(1)[gids > 0], counts.sum(0)[pids > 0]
    inter = counts[np.ix_(gids > 0, pids > 0)]
    gids, pids = gids[gids > 0], pids[pids > 0]
    gfrac = inter / np.maximum(ga[:, None], 1)
    pfrac = inter / np.maximum(pa[None, :], 1)
    graph = (gfrac >= threshold) & (pfrac >= threshold)
    split_ids, merge_ids = gids[graph.sum(1) >= 2], pids[graph.sum(0) >= 2]
    merge_gt_ids = gids[graph[:, graph.sum(0) >= 2].any(1)]
    gt_uncovered = gids[(ga-inter.sum(1))/np.maximum(ga, 1) > .5]
    gt_nearly_uncovered = gids[inter.sum(1)/np.maximum(ga, 1) < .1]
    gt_fully_uncovered = gids[inter.sum(1) == 0]
    pred_extra = pids[(pa-inter.sum(0))/np.maximum(pa, 1) > .5]

    # Recover nearby GT interfaces across the eroded annotation's background gap.
    expanded = gt.copy()
    if gids.size:
        distance, indices = ndi.distance_transform_edt(gt == 0, return_indices=True)
        close = (gt == 0) & (distance <= neighbor_distance)
        expanded[close] = gt[tuple(indices[:, close])]
    pairs = []
    for a, b in ((expanded[:-1], expanded[1:]), (expanded[:, :-1], expanded[:, 1:])):
        adjacent = (a > 0) & (b > 0) & (a != b)
        pairs.append(np.sort(np.stack([a[adjacent], b[adjacent]], axis=1), axis=1))
    adjacent_pairs = np.unique(np.concatenate(pairs), axis=0)
    dominant = np.zeros(len(gids), dtype=int)
    if pids.size:
        maximum = inter.argmax(1)
        covered = gfrac[np.arange(len(gids)), maximum] >= .5
        dominant[covered] = pids[maximum[covered]]
    missing_pairs = []
    for a, b in adjacent_pairs:
        pa_id, pb_id = dominant[np.searchsorted(gids, [a, b])]
        if pa_id > 0 and pa_id == pb_id:
            missing_pairs.append([int(a), int(b), int(pa_id)])
    return dict(gt_count=len(gids), pred_count=len(pids),
                split_gt_ids=split_ids.tolist(), merged_pred_ids=merge_ids.tolist(),
                gt_in_merge_ids=merge_gt_ids.tolist(),
                majority_uncovered_gt_ids=gt_uncovered.tolist(),
                nearly_uncovered_gt_ids=gt_nearly_uncovered.tolist(),
                fully_uncovered_gt_ids=gt_fully_uncovered.tolist(),
                majority_extra_pred_ids=pred_extra.tolist(),
                split_gt_rate=float(len(split_ids)/len(gids)) if len(gids) else 0.,
                merged_pred_rate=float(len(merge_ids)/len(pids)) if len(pids) else 0.,
                gt_in_merge_rate=float(len(merge_gt_ids)/len(gids)) if len(gids) else 0.,
                neighbor_gt_pairs=len(adjacent_pairs),
                neighboring_pairs_in_same_prediction=missing_pairs,
                overlap_threshold=threshold, neighbor_expansion_pixels=neighbor_distance)


def boundary_distances(truth, prediction, margin=10):
    """Four-neighbor label boundaries and Euclidean distances, excluding frame band.

    These thin label-transition boundaries differ from the paper's dilated metric.
    Both boundaries and the distance targets exclude the frame band.
    """
    gt, pred = _labels(truth), _labels(prediction)
    if gt.shape != pred.shape or min(gt.shape) <= 2*margin or margin < 0:
        raise ValueError('Invalid shapes or frame margin')
    gb, pb = boundaries(gt), boundaries(pred)
    if margin:
        for mask in (gb, pb):
            mask[:margin] = mask[-margin:] = False
            mask[:, :margin] = mask[:, -margin:] = False
    to_pred = ndi.distance_transform_edt(~pb) if pb.any() else np.full(gt.shape, np.inf)
    to_gt = ndi.distance_transform_edt(~gb) if gb.any() else np.full(gt.shape, np.inf)
    return gb, pb, to_pred, to_gt


def boundary_distance_summary(truth, prediction, tolerances=(0, 1, 2, 3, 5, 10), margin=10):
    gb, pb, dp, dg = boundary_distances(truth, prediction, margin)
    rows = []
    for tol in tolerances:
        precision = float((dg[pb] <= tol).mean()) if pb.any() else float(not gb.any())
        recall = float((dp[gb] <= tol).mean()) if gb.any() else float(not pb.any())
        rows.append(dict(tolerance_pixels=tol, precision=precision, recall=recall,
                         f1=2*precision*recall/(precision+recall) if precision+recall else 0.))
    return dict(curve=rows, frame_margin_pixels=margin,
                gt_boundary_within_3px=float((dp[gb] <= 3).mean()) if gb.any() else 1.,
                gt_boundary_3_to_10px=float(((dp[gb] > 3) & (dp[gb] <= 10)).mean()) if gb.any() else 0.,
                gt_boundary_beyond_10px=float((dp[gb] > 10).mean()) if gb.any() else 0.,
                gt_to_pred_median_px=float(np.median(dp[gb])) if gb.any() and pb.any() else None,
                gt_to_pred_p90_px=float(np.quantile(dp[gb], .9)) if gb.any() and pb.any() else None)


def bootstrap_mean(values, resample_indices):
    values = np.asarray(values, dtype=float)
    indices = np.asarray(resample_indices)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('Finite per-image values required')
    if indices.ndim != 2 or indices.shape[1] != len(values):
        raise ValueError('Bootstrap rows must each resample all image positions')
    means = values[indices].mean(1)
    lo, hi = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), ci95_low=float(lo), ci95_high=float(hi),
                images=len(values), resamples=len(indices))


def paired_mean_comparison(left, right, resample_indices):
    """Pair-preserving bootstrap, paired t interval, and exact sign-flip check.

    The t interval assumes a sufficiently regular distribution of paired errors.
    The sign-flip reference requires sign exchangeability under the null. Neither
    addresses uncertainty from refitting a model. Exact enumeration is capped.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError('At least two matched image pairs required')
    difference = left-right
    result = bootstrap_mean(difference, resample_indices)
    halfwidth = float(student_t.ppf(.975, len(left)-1)*difference.std(ddof=1)/np.sqrt(len(left)))
    result.update(paired_t_ci95_low=result['mean']-halfwidth,
                  paired_t_ci95_high=result['mean']+halfwidth)
    if len(left) <= 16:
        signs = 2*((np.arange(1<<len(left))[:, None] >> np.arange(len(left))) & 1)-1
        null_means = (signs*difference).mean(1)
        result['exact_sign_flip_p_two_sided'] = float(np.mean(abs(null_means) >= abs(result['mean'])-1e-14))
    else:
        result['exact_sign_flip_p_two_sided'] = None
    return result
