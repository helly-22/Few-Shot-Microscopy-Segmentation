# Controlled Voronoi experiment (H1)

The native Cellpose-SAM runner is the reproduction baseline. This document
describes the later geometry experiment, which is not implemented end to end.

## Matched conditions

Compare an identical architecture trained with its supervised objective against
the same model plus `lambda_voronoi * voronoi_consistency_loss(...)`. Match initial
weights, trainable parameters, seed supervision, training examples, augmentations,
optimizer, update count, hyperparameter search budget, and inference procedure.
A frozen encoder with lightweight heads is a possible H1 architecture; it differs
from the current full-network baseline and needs its own matched control.

`geometry.py` assumes aligned instance channels and predicted seeds; it does not
infer the number of objects or provide background segmentation. A trainable adapter
must implement those components and channel matching. Do not use ground-truth
centers at deployment; label any oracle-center experiment as diagnostic.

## Few-shot splits

Shots count full annotated images, not cropped patches or individual grains.
Use `make_experiments` in `splits.py` to construct paired 2/5-shot subsets across
seeds, with fixed validation and test specimens. Assign specimens before cropping.
Where physical specimen IDs are unavailable, micrograph-level grouping prevents
patch leakage but cannot establish specimen-independent generalization.

The paper-reconstruction runner uses a separate fixed split protocol and does
not yet implement this H1 multi-seed trainer. Select lambda and temperature using
validation data, never test masks. Synthetic masks test geometry correctness;
performance on generated Voronoi shapes alone is not evidence supporting H1.

## Evaluation

Report false-split and false-merge rates, boundary F1, PQ, matched Dice, and object
counts. The H1 helpers use nonnegative integer masks with zero as background.
Zero is not an ignore label; handle excluded areas explicitly before evaluation.

The generic H1 boundary metric uses one-pixel Manhattan tolerance by default.
Split/merge overlaps must cover at least 10% of both objects by default. Declare
thresholds before evaluation and examine sensitivity. Matched Dice excludes
unmatched objects, so interpret it with PQ. These conventions differ from the
paper-style metrics in `baseline.py`.

Predeclare lower split rate and higher boundary F1 as outcomes, with a numerical
acceptable increase in merge rate. Aggregate images within specimens, compute
paired specimen-level differences and confidence intervals, and report variation
across training seeds and few-shot subsets. Individual grains and repeated seeds
are not independent test specimens. H2 measurements and H3 biological transfer
are separate follow-up experiments.
