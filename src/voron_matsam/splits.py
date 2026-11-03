"""Fixed specimen holdouts and paired few-shot subsets; no patch leakage."""

import random


def make_experiments(records, validation_specimens, test_specimens,
                     shots=(2, 5), seeds=(0, 1, 2, 3, 4)):
    """Records contain image_id and specimen_id. Shot counts refer to images.

    All images from held-out specimens are excluded from the training pool.
    Each seed uses nested subsets and identical records for both conditions.
    """
    val, test = set(validation_specimens), set(test_specimens)
    if not val or not test or val & test:
        raise ValueError("validation/test specimens must be nonempty and disjoint")
    ids = [r["image_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("image_id must be unique")
    known = {r["specimen_id"] for r in records}
    if not (val | test) <= known:
        raise ValueError("unknown held-out specimen")
    pool = sorted(r["image_id"] for r in records if r["specimen_id"] not in val | test)
    shots, seeds = tuple(shots), tuple(seeds)
    if not shots or any(not isinstance(k, int) or k < 1 or k > len(pool) for k in shots):
        raise ValueError("shots must be positive and fit the training pool")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("provide unique seeds")
    shared = {"validation": sorted(r["image_id"] for r in records if r["specimen_id"] in val),
              "test": sorted(r["image_id"] for r in records if r["specimen_id"] in test)}
    runs = []
    for seed in seeds:
        sample = random.Random(seed).sample(pool, len(pool))
        for k in shots:
            for condition in ("baseline", "voronoi"):
                runs.append(dict(seed=seed, shots=k, condition=condition,
                                 train=sample[:k], **shared))
    return runs
