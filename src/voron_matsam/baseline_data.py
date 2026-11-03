"""Prepare only real RG/RGMask data from Warren's Kaggle v4 archive."""

import hashlib
import io
import json
from pathlib import Path
import random
import re
import urllib.request
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

DATA_URL = ("https://www.kaggle.com/api/v1/datasets/download/"
            "peterwarren/voronoi-artificial-grains-gen?datasetVersionNumber=4")
FRACTIONS = {"05": 2, "10": 4, "25": 10, "50": 20, "75": 30}
DATA_SHA256 = "514d213abb0ce7c13da6d8063256061aea763e1fe959e5251278dd86be3836c6"
MODEL_REVISION = "7c61431b5fbb078f3296754bd15d9f51b320f837"
MODEL_SHA256 = "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2"
MODEL_URL = f"https://huggingface.co/mouseland/cellpose-sam/resolve/{MODEL_REVISION}/cpsam"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download_data(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Already exists: {destination}; use prepare to verify it")
    temporary = destination.with_suffix(".download")
    urllib.request.urlretrieve(DATA_URL, temporary)
    if not zipfile.is_zipfile(temporary):
        raise ValueError("Dataset response was not a ZIP; check Kaggle access")
    if sha256(temporary) != DATA_SHA256:
        raise ValueError("Dataset archive differs from audited v4 download; inspect before updating the pin")
    temporary.replace(destination)
    print(f"Downloaded {destination}; sha256={sha256(destination)}")


def download_weights(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == MODEL_SHA256:
        print(f"Verified existing checkpoint: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Checkpoint exists but checksum differs: {destination}")
    temporary = destination.with_suffix(".download")
    urllib.request.urlretrieve(MODEL_URL, temporary)
    if sha256(temporary) != MODEL_SHA256:
        raise ValueError("Checkpoint checksum mismatch")
    temporary.replace(destination)
    print(f"Downloaded and verified checkpoint: {destination}")


def disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def compact_labels(labels):
    """Keep 0 as background even when an image contains no zero pixels."""
    ids = np.unique(labels)
    ids = ids[ids > 0]
    result = np.zeros(labels.shape, dtype=np.uint32)
    positive = labels > 0
    result[positive] = np.searchsorted(ids, labels[positive]) + 1
    return result


def boundary_to_instances(mask, polarity="bright", min_size=200):
    if polarity not in ("bright", "dark"):
        raise ValueError("polarity must be bright or dark")
    interior = mask >= 128 if polarity == "bright" else mask < 128
    # Match released code: label first, then erode each original component.
    initial, _ = ndi.label(interior)
    separated = np.zeros_like(initial)
    for region, bounds in enumerate(ndi.find_objects(initial), start=1):
        if bounds is None:
            continue
        # Pad bounding boxes to reproduce a full-image erosion at every edge.
        piece = initial[bounds] == region
        padded = np.pad(piece, 1)
        # skimage 0.24 binary_erosion defaults to mode='ignore': outside the
        # full image is True. Outside an internal object bounding box is False.
        for axis, bound in enumerate(bounds):
            if bound.start == 0:
                edge = [slice(None)] * 2
                edge[axis] = 0
                padded[tuple(edge)] = True
            if bound.stop == initial.shape[axis]:
                edge = [slice(None)] * 2
                edge[axis] = -1
                padded[tuple(edge)] = True
        eroded = ndi.binary_erosion(padded, structure=disk(1))[1:-1, 1:-1]
        if eroded.sum() >= min_size:
            out = separated[bounds]
            out[eroded] = region
    return compact_labels(separated)


def archive_groups(archive):
    patterns = {
        "image": re.compile(r"^GRAIN DATA SET/RG/RG(\d+)_(\d+)_(\d+)\.jpg$"),
        "mask": re.compile(r"^GRAIN DATA SET/RGMask/RGMask(\d+)_(\d+)_(\d+)\.jpg$"),
    }
    groups = {}
    for name in archive.namelist():
        for kind, pattern in patterns.items():
            match = pattern.fullmatch(name)
            if match:
                sid, row, col = match.groups()
                tiles = groups.setdefault(f"RG{sid}", {"image": {}, "mask": {}})[kind]
                coordinate = (int(row), int(col))
                if coordinate in tiles:
                    raise ValueError(f"Duplicate tile {name}")
                tiles[coordinate] = name
    expected = {(r, c) for r in range(1, 4) for c in range(1, 5)}
    if len(groups) != 40:
        raise ValueError(f"Expected 40 real micrographs, found {len(groups)}")
    for sid, pair in groups.items():
        if any(set(pair[kind]) != expected for kind in pair):
            raise ValueError(f"Incomplete or mismatched 3x4 tile grid: {sid}")
    return groups


def stitch(archive, tiles, mode):
    arrays = {}
    shape = None
    for coordinate, name in sorted(tiles.items()):
        with Image.open(io.BytesIO(archive.read(name))) as image:
            array = np.array(image.convert(mode))
        if shape is not None and array.shape != shape:
            raise ValueError(f"Tile dimensions differ: {name}")
        shape = array.shape
        arrays[coordinate] = array
    return np.concatenate([np.concatenate([arrays[r, c] for c in range(1, 5)], axis=1)
                           for r in range(1, 4)], axis=0)


def prepare_dataset(archive_path, output, polarity="bright", split_seed=42):
    import tifffile

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix prepared datasets in {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("images", "instances", "raw_masks"):
        (output / name).mkdir()
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        groups = archive_groups(archive)
        for sid in sorted(groups):
            image = stitch(archive, groups[sid]["image"], "RGB")
            mask = stitch(archive, groups[sid]["mask"], "L")
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions differ for {sid}")
            labels = boundary_to_instances(mask, polarity)
            if labels.max() < 5:
                raise ValueError(f"Only {labels.max()} objects in {sid}; inspect mask polarity")
            image_path, mask_path = f"images/{sid}_img.png", f"instances/{sid}_masks.tif"
            raw_path = f"raw_masks/{sid}_mask.png"
            Image.fromarray(image).save(output / image_path)
            Image.fromarray(mask).save(output / raw_path)
            tifffile.imwrite(output / mask_path, labels)
            records.append(dict(image_id=sid, image=image_path, mask=mask_path,
                                raw_mask=raw_path, instances=int(labels.max()),
                                shape=list(mask.shape),
                                sha256_image=sha256(output / image_path),
                                sha256_mask=sha256(output / mask_path),
                                sha256_raw_mask=sha256(output / raw_path)))
    order = [r["image_id"] for r in records]
    random.Random(split_seed).shuffle(order)
    manifest = dict(schema_version=1, source_url=DATA_URL,
                    source_archive_sha256=sha256(archive_path), polarity=polarity,
                    threshold=128, erosion_radius=1, min_area=200,
                    split_seed=split_seed, split_status="reconstructed_not_author_confirmed",
                    grouping="source micrograph; physical specimen IDs unavailable",
                    train_order=order[:30], test=order[30:],
                    fractions={key: order[:n] for key, n in FRACTIONS.items()}, records=records)
    write_json(output / "manifest.json", manifest)
    print(f"Prepared {len(records)} micrographs; 30 training pool / 10 test")
    return manifest


def load_manifest(path, verify=True):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = {r["image_id"]: r for r in manifest["records"]}
    if len(records) != len(manifest["records"]):
        raise ValueError("Duplicate image IDs")
    train_ids, test_ids = manifest["train_order"], manifest["test"]
    if (not test_ids or len(set(train_ids + test_ids)) != len(train_ids + test_ids)
            or set(train_ids + test_ids) != set(records)):
        raise ValueError("Invalid training/test partition")
    for key, subset in manifest["fractions"].items():
        if subset != train_ids[:len(subset)] or set(subset) & set(test_ids):
            raise ValueError(f"Non-nested/leaking split {key}")
    for record in records.values():
        for key in ("image", "mask", "raw_mask"):
            full = (path.parent / record[key]).resolve()
            if path.parent not in full.parents:
                raise ValueError("Dataset path escapes manifest directory")
            if verify and sha256(full) != record[f"sha256_{key}"]:
                raise ValueError(f"Dataset file changed: {full}")
    return manifest, records, path.parent
