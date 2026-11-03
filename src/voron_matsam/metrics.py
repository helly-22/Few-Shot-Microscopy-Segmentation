"""Explicit, ID-invariant instance metrics for 2D integer label maps.

Zero is background. Split/merge overlap requires at least 10% of BOTH objects
by default; report the threshold with results. Boundaries exclude image edges.
"""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment


def _labels(mask):
    a = np.asarray(mask)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) or (a < 0).any():
        raise ValueError("masks must be nonnegative 2D integer labels")
    return a


def boundaries(mask):
    a = _labels(mask)
    b = np.zeros(a.shape, dtype=bool)
    d = a[1:] != a[:-1]
    b[1:] |= d
    b[:-1] |= d
    d = a[:, 1:] != a[:, :-1]
    b[:, 1:] |= d
    b[:, :-1] |= d
    return b


def evaluate_instances(truth, prediction, overlap_threshold=0.1, boundary_tolerance=1):
    """Return PQ, matched Dice, split/merge rates, and tolerant boundary F1.

    PQ uses one-to-one matches with IoU > 0.5. Matched Dice excludes unmatched
    objects, so always interpret it alongside PQ. Split rate denominator is GT
    objects; merge rate denominator is predicted objects. Empty pairs score 1
    for agreement metrics and 0 for error rates.
    """
    gt, pred = _labels(truth), _labels(prediction)
    if gt.shape != pred.shape or gt.size == 0:
        raise ValueError("masks must have identical nonempty shapes")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    if not isinstance(boundary_tolerance, int) or boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be a nonnegative integer")
    gids, gi = np.unique(gt, return_inverse=True)
    pids, pi = np.unique(pred, return_inverse=True)
    table = np.bincount(gi.ravel() * len(pids) + pi.ravel(),
                        minlength=len(gids) * len(pids)).reshape(len(gids), len(pids))
    ga = table.sum(1)[gids > 0]
    pa = table.sum(0)[pids > 0]
    inter = table[np.ix_(gids > 0, pids > 0)]
    union = ga[:, None] + pa[None, :] - inter
    iou = inter / np.maximum(union, 1)
    rows, cols = linear_sum_assignment(-iou)
    keep = iou[rows, cols] > 0.5
    rows, cols = rows[keep], cols[keep]
    denom = 0.5 * (len(ga) + len(pa))
    pq = float(iou[rows, cols].sum() / denom) if denom else 1.0
    dice = (float((2 * inter[rows, cols] / (ga[rows] + pa[cols])).mean())
            if len(rows) else float(denom == 0))
    significant = ((inter / np.maximum(ga[:, None], 1) >= overlap_threshold)
                   & (inter / np.maximum(pa[None, :], 1) >= overlap_threshold))
    split = float((significant.sum(1) > 1).mean()) if len(ga) else 0.0
    merge = float((significant.sum(0) > 1).mean()) if len(pa) else 0.0
    gb, pb = boundaries(gt), boundaries(pred)
    # scipy iterations=0 dilates until convergence, so handle zero explicitly.
    gd = binary_dilation(gb, iterations=boundary_tolerance) if boundary_tolerance else gb
    pd = binary_dilation(pb, iterations=boundary_tolerance) if boundary_tolerance else pb
    precision = float(gd[pb].mean()) if pb.any() else float(not gb.any())
    recall = float(pd[gb].mean()) if gb.any() else float(not pb.any())
    bf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(pq=pq, matched_dice=dice, over_segmentation_rate=split,
                under_segmentation_rate=merge, boundary_f1=bf1,
                gt_instances=len(ga), predicted_instances=len(pa))
