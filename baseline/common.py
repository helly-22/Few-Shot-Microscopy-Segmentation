"""Shared splits, image normalization, watershed conversion and evaluation."""

import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import tifffile
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed

from voron_matsam.baseline_data import compact_labels, load_manifest, sha256, write_json
from voron_matsam.metrics import evaluate_instances

METHODS = ('cellpose_sam', 'cellpose2', 'cellpose3', 'unet_watershed', 'sam2', 'stardist', 'watershed')


def normalize(image):
    image = np.asarray(image, dtype=np.float32)
    axes = (0, 1)
    lo, hi = np.percentile(image, (1, 99), axis=axes, keepdims=True)
    return np.clip((image - lo) / np.maximum(hi - lo, 1e-6), 0, 1).astype(np.float32)


def read_pair(record, root):
    with Image.open(root / record['image']) as image:
        x = np.array(image.convert('RGB'))
    y = tifffile.imread(root / record['mask'])
    if x.shape[:2] != y.shape:
        raise ValueError('Image and label dimensions differ')
    return x, compact_labels(y)


def make_plan(manifest_path, output, seeds=(0, 1, 2, 3, 4), shots=(2, 5), validation_count=5):
    manifest, records, root = load_manifest(manifest_path)
    pool = manifest['train_order']
    if not 1 <= validation_count < len(pool):
        raise ValueError('Reserve at least one validation and one training image')
    validation = pool[-validation_count:]
    training = pool[:-validation_count]
    if not shots or any(k < 1 or k > len(training) for k in shots):
        raise ValueError('Shot counts must fit the training pool')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide unique training seeds')
    runs = []
    for seed in seeds:
        order = random.Random(seed).sample(training, len(training))
        for shot in shots:
            runs.append(dict(seed=seed, shots=shot, train=order[:shot],
                             validation=validation, test=manifest['test']))
    plan = dict(protocol='baseline_suite_v1', manifest_sha256=sha256(manifest_path),
                grouping=manifest.get('grouping', 'source micrograph'), runs=runs)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    return plan


def select_run(plan_path, manifest_path, shots, seed):
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan['manifest_sha256'] != sha256(manifest_path):
        raise ValueError('Plan belongs to a different or modified manifest')
    matches = [r for r in plan['runs'] if r['shots'] == shots and r['seed'] == seed]
    if len(matches) != 1:
        raise ValueError('Need exactly one plan entry for this shot count and seed')
    run = matches[0]
    partitions = [run[k] for k in ('train', 'validation', 'test')]
    if any(not group for group in partitions):
        raise ValueError('Training, validation and test partitions must be nonempty')
    flat = sum(partitions, [])
    if len(set(flat)) != len(flat) or len(run['train']) != shots:
        raise ValueError('Duplicate/leaking IDs or incorrect shot count')
    return run


def distance_instances(foreground, min_distance=12, min_area=200):
    """Marker-controlled distance watershed; never needs instance labels."""
    foreground = np.asarray(foreground, dtype=bool)
    distance = ndi.distance_transform_edt(foreground)
    points = peak_local_max(distance, min_distance=min_distance,
                            labels=foreground.astype(np.uint8), exclude_border=False)
    markers = np.zeros(foreground.shape, dtype=np.int32)
    if len(points):
        markers[tuple(points.T)] = np.arange(1, len(points) + 1)
    # Guarantee a marker in every disconnected foreground component.
    components, n = ndi.label(foreground)
    next_id = int(markers.max())
    for label, bounds in enumerate(ndi.find_objects(components), 1):
        component = components[bounds] == label
        if np.any(markers[bounds][component]):
            continue
        local = np.where(component, distance[bounds], -1)
        position = np.unravel_index(local.argmax(), local.shape)
        index = tuple(position[a] + bounds[a].start for a in range(2))
        next_id += 1
        markers[index] = next_id
    labels = watershed(-distance, markers, mask=foreground)
    sizes = np.bincount(labels.ravel())
    labels[sizes[labels] < min_area] = 0
    return compact_labels(labels)


def classical_instances(image, config):
    gray = normalize(image).mean(axis=-1)
    gray = gaussian(gray, sigma=config['sigma'], preserve_range=True)
    if np.ptp(gray) == 0:
        return np.zeros(gray.shape, dtype=np.uint32)
    threshold = threshold_otsu(gray) + config['threshold_offset']
    foreground = gray >= threshold if config['bright_interiors'] else gray < threshold
    return distance_instances(foreground, config['min_distance'], config['min_area'])


def prompt_points(image, config):
    """Derive XY prompt coordinates from the image, with a fixed prompt budget."""
    labels = classical_instances(image, config)
    points = []
    for gid, bounds in enumerate(ndi.find_objects(labels), 1):
        if bounds is None:
            continue
        region = labels[bounds] == gid
        distances = ndi.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        y, x = np.unravel_index(distances.argmax(), distances.shape)
        points.append((float(distances[y, x]), int(x + bounds[1].start), int(y + bounds[0].start)))
    points.sort(key=lambda p: (-p[0], p[2], p[1]))
    return np.asarray([(x, y) for _, x, y in points[:config['max_prompts']]], dtype=np.float32).reshape(-1, 2)


def ap_metrics(truth, prediction):
    """Cellpose-style TP/(TP+FP+FN); not COCO confidence-ranked average precision."""
    gt, pred = compact_labels(truth), compact_labels(prediction)
    ng, npred = int(gt.max()), int(pred.max())
    table = np.bincount((gt.astype(np.int64) * (npred + 1) + pred).ravel(),
                        minlength=(ng + 1) * (npred + 1)).reshape(ng + 1, npred + 1)
    inter = table[1:, 1:]
    union = table[1:, :].sum(1)[:, None] + table[:, 1:].sum(0)[None, :] - inter
    iou = inter / np.maximum(union, 1)
    scores = []
    for threshold in np.linspace(.5, .95, 10):
        eligible = iou >= threshold
        cost = -eligible.astype(float) - iou / (2 * max(ng, npred, 1))
        rows, cols = linear_sum_assignment(cost)
        tp = int(eligible[rows, cols].sum())
        denominator = ng + npred - tp
        scores.append(tp / denominator if denominator else 1.)
    return dict(ap50=scores[0], map_50_95=float(np.mean(scores)))


def evaluate(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError('Prediction and reference shapes differ')
    if not np.issubdtype(prediction.dtype, np.integer) or (prediction < 0).any():
        raise ValueError('Predictions must be nonnegative integer instances')
    metrics = evaluate_instances(truth, prediction, overlap_threshold=.1, boundary_tolerance=1)
    metrics.update(ap_metrics(truth, prediction))
    metrics['count_error'] = metrics['predicted_instances'] - metrics['gt_instances']
    return metrics
