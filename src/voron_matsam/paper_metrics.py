"""Common, model-independent reconstruction of the released paper metrics.

AP is Cellpose matching accuracy, not confidence-ranked COCO AP. Boundary F1
uses outer boundaries, disk(2) dilation and disk(3) matching tolerance.
"""

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries

from .baseline_data import compact_labels, disk


def score_instances(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.ndim != 2 or truth.shape != prediction.shape:
        raise ValueError("Need matching 2D instance masks")
    for mask in (truth, prediction):
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError("Instance masks must contain nonnegative integers")
    gt, pred = compact_labels(truth), compact_labels(prediction)
    ng, npred = int(gt.max()), int(pred.max())
    table = np.bincount((gt.astype(np.int64) * (npred + 1) + pred).ravel(),
                        minlength=(ng + 1) * (npred + 1)).reshape(ng + 1, npred + 1)
    intersection = table[1:, 1:]
    union = table[1:, :].sum(1)[:, None] + table[:, 1:].sum(0)[None, :] - intersection
    iou = intersection / np.maximum(union, 1)
    scores = []
    for threshold in np.linspace(.5, .95, 10):
        eligible = iou >= threshold
        rows, cols = linear_sum_assignment(-eligible.astype(float) - iou / (2 * max(ng, npred, 1)))
        tp = int(eligible[rows, cols].sum())
        denominator = ng + npred - tp
        # Explicit empty/empty convention; this case does not occur in the data.
        scores.append(tp / denominator if denominator else 1.)
    gt_bd = ndi.binary_dilation(find_boundaries(gt, mode="outer"), structure=disk(2))
    pred_bd = ndi.binary_dilation(find_boundaries(pred, mode="outer"), structure=disk(2))
    gt_tol = ndi.binary_dilation(gt_bd, structure=disk(3))
    pred_tol = ndi.binary_dilation(pred_bd, structure=disk(3))
    precision = np.logical_and(pred_bd, gt_tol).sum() / (pred_bd.sum() + 1e-7)
    recall = np.logical_and(gt_bd, pred_tol).sum() / (gt_bd.sum() + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    binary = (pred > 0) & ~pred_bd
    return dict(ap50=float(scores[0]), map_50_95=float(np.mean(scores)),
                boundary_f1=float(f1), count_error=npred - ng,
                gt_count=ng, pred_count=npred), binary
