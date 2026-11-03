"""Reconstruction of the released circular Jeffries evaluation, with diagnostics."""

import numpy as np
from scipy import ndimage as ndi


def radial_extents(binary):
    if binary.ndim != 2 or min(binary.shape) < 3:
        raise ValueError("Need a nonempty 2D image")
    labels, n = ndi.label(binary, structure=np.ones((3, 3)))
    h, w = labels.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ids = np.arange(1, n + 1)
    minimum = ndi.minimum(distances, labels, ids) if n else np.array([])
    maximum = ndi.maximum(distances, labels, ids) if n else np.array([])
    return np.asarray(minimum), np.asarray(maximum), min(cx, cy)


def measure(extents, target=60, pixels_per_um=2.26, radius=None):
    minimum, maximum, limit = extents
    if target < 1 or not np.isfinite(pixels_per_um) or pixels_per_um <= 0:
        raise ValueError("Positive target and finite calibration required")
    if radius is None:
        radius = next((r for r in range(10, int(limit), 2)
                       if np.count_nonzero(maximum < r) >= target), limit)
    if not 0 < radius <= limit:
        raise ValueError("Circle radius outside image")
    internal = int(np.count_nonzero(maximum < radius))
    intercepted = int(np.count_nonzero((minimum <= radius) & (maximum >= radius)))
    area = np.pi * radius ** 2 / pixels_per_um ** 2 / 1e6
    density = (internal + intercepted / 2) / area
    return dict(radius_px=float(radius), internal=internal, intercepted=intercepted,
                area_mm2=float(area), na=float(density),
                g=float(3.321928 * np.log10(density) - 2.954) if density > 0 else None,
                target_met=internal >= target)


def mape(truth, prediction):
    """Never silently drop zero/negative/undefined references."""
    gt, pred = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if gt.size == 0 or not np.isfinite(gt).all() or not np.isfinite(pred).all() or (gt <= 0).any():
        return None
    return float(np.mean(np.abs((pred - gt) / gt)) * 100)


def evaluate_pair(gt_binary, pred_binary, pixels_per_um=2.26, targets=range(10, 101, 10)):
    if gt_binary.shape != pred_binary.shape:
        raise ValueError("Jeffries masks must have identical shape")
    gt_extents, pred_extents = radial_extents(gt_binary), radial_extents(pred_binary)
    rows = []
    for target in targets:
        gt = measure(gt_extents, target, pixels_per_um)
        for mode in ("superimposed", "independent"):
            pred = measure(pred_extents, target, pixels_per_um,
                           gt["radius_px"] if mode == "superimposed" else None)
            rows.append(dict(target=target, mode=mode, gt=gt, pred=pred))
    return rows
