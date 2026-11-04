"""Measure actual saved tensor changes against dtype-matched initial weights.

This is a descriptive audit, not proof that precision caused a metric gap.
Run checkpoint loading in a CPU allocation, not on the cluster login node.
"""

import argparse
from pathlib import Path

import torch

from voron_matsam.baseline_data import sha256, write_json


def audit(initial, trained):
    rows, skipped = [], []
    for name, value in trained.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if name not in initial or initial[name].shape != value.shape:
            skipped.append(name)
            continue
        # Loading initial fp32 weights into a bf16 model already rounds them.
        # Compare against that rounded initial state, rather than counting the
        # initial cast itself as learned weight changes.
        original = initial[name].to(value.dtype).reshape(-1)
        current = value.reshape(-1)
        changed, absolute, maximum = 0, 0., 0.
        for start in range(0, current.numel(), 1_000_000):
            a, b = original[start:start + 1_000_000], current[start:start + 1_000_000]
            changed += int(torch.count_nonzero(a != b))
            delta = (a.float() - b.float()).abs()
            absolute += float(delta.double().sum())
            maximum = max(maximum, float(delta.max()) if delta.numel() else 0.)
        count = current.numel()
        rows.append(dict(name=name, initial_dtype=str(initial[name].dtype),
                         trained_dtype=str(value.dtype), elements=count, changed=changed,
                         fraction_changed=changed / count if count else 0.,
                         mean_absolute_update=absolute / count if count else 0.,
                         max_absolute_update=maximum))
    count = sum(row['elements'] for row in rows)
    changed = sum(row['changed'] for row in rows)
    return dict(tensors=rows, unmatched_trained_keys=skipped,
                missing_trained_keys=sorted(set(initial) - set(trained)),
                elements=count, changed=changed,
                fraction_changed=changed / count if count else None,
                exactly_unchanged_tensors=sum(row['changed'] == 0 for row in rows),
                total_compared_tensors=len(rows),
                comparison='initial cast to each trained tensor dtype before comparison',
                interpretation_limit='Saved weight differences do not identify a causal mechanism; '
                    'need a matched precision control and validation evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--trained', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    initial = torch.load(args.initial, map_location='cpu', weights_only=True)
    trained = torch.load(args.trained, map_location='cpu', weights_only=True)
    report = audit(initial, trained)
    report.update(initial_sha256=sha256(args.initial), trained_sha256=sha256(args.trained),
                  source_sha256=sha256(__file__), torch_version=torch.__version__)
    write_json(output, report)
    print({key: value for key, value in report.items() if key != 'tensors'})


if __name__ == '__main__':
    main()
