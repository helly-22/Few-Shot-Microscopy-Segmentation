# Inspect the two shot baseline before testing Voronoi constraints

The diagnostic reads saved predictions from the final 2-shot checkpoint. It does
not retrain a model or choose checkpoints using test scores. The one available
2-shot run happens to have the highest observed AP50 in this seed; it is not a
validation-selected best checkpoint.

## Inputs and execution

Run `scripts/collect_diagnostic_inputs.py` on Unity, where the checkpoint and the
prepared dataset exist. Use a new archive filename:

```bash
.hpc-env/bin/python scripts/collect_diagnostic_inputs.py \
  --run outputs/paper-20261103T234304Z/baselines/05_seed0 \
  --output outputs/two_shot_diagnostic_inputs.tar.gz
```

The collector verifies the checkpoint and manifest hashes, then collects the
exact ten test microscopy images, integer GT masks, raw GT masks, and saved
instance and binary predictions. Keep the archive and all outputs out of Git.
Unpack this archive under `outputs/two_shot_diagnostic/inputs`. The completed
per-image results and run metadata for all six configurations must also be
available under `outputs/paper-20261103T234304Z/baselines`.

The measurement cross-check uses the authors' source at revision
`80f305e9ae10d3f845914f6d773ecd2f8fcea651`, under
`external/ASTM-Grain-Size-Estimator/4_evaluate_jeffries.py`.
Install the diagnostic extras in your analysis environment if needed:

```bash
python -m pip install -e '.[diagnostic]'
python scripts/diagnose_two_shot.py
python scripts/finalize_two_shot_diagnostic.py
```

Windows local invocation: `.\.venv\Scripts\python.exe scripts/diagnose_two_shot.py`.
Use `--skip-plots` to rebuild scores and the report while retaining existing image
panels. The script reads a curated `visual_review.json` when present; those notes
are human visual observations, not automated labels of the physical mechanism.

## What the outputs mean

`diagnostic_report.html` links all ten four-panel figures, original microscopy,
GT instances, predicted instances, and an overlay. Orange marks split GT
candidates and magenta marks merged predicted candidates. Colors in the GT and
predicted instance panels are local IDs, not matched identities. Supplemental
boundary images show GT support within 3 pixels, beyond 3 through 10 pixels, and
beyond 10 pixels of a predicted boundary.

The main split/merge graph requires intersection area to cover at least 10% of
both objects. A GT node with at least two predicted neighbors is a split
candidate; a predicted node with at least two GT neighbors is a merge candidate.
Events can coexist and can be caused by contour displacement or limited bleed
across a neighboring object. The sensitivity CSV repeats the counts at 5% and
20%. Denominators are GT objects for splits and predicted objects for merges.

Majority uncovered GT has over 50% of its area predicted as background. Nearly
uncovered GT has less than 10% foreground coverage; fully uncovered GT has zero
coverage. These distinguish missed or contracted foreground from true mergers.
The nearby-pair check expands GT labels by at most 5 pixels across annotation
gaps, finds 4-connected neighbors, and asks whether both have at least 50% of
their area in the same dominant predicted label. It is a conservative check,
not a complete annotation of all possible neighboring-grain merges.

The boundary distance diagnostic uses four-neighbor label transitions and
Euclidean nearest-boundary distances, with a 10-pixel frame band excluded. It
differs from the paper's dilated boundary F1, which is preserved in the per-image
score table. Unsupported within 10 pixels is not proof of a missing physical
boundary; more severe displacement and missing foreground are alternatives.

## Uncertainty and comparability

The bootstrap resamples whole image positions, using the same positions for each
model in a paired difference. It uses 20000 draws, seed 20261104, and 95%
percentile intervals. The paired difference table also includes a paired t
interval with 9 degrees of freedom and a two-sided exact sign-flip reference
over all 1024 sign patterns. The latter assumes sign exchangeability under the
null. Methods can disagree with ten images; do not select only the favorable
interval. Comparisons are exploratory and unadjusted for multiplicity.

These calculations condition on one trained model per split. They do not capture
training-seed variability, and physical specimen independence is unverified.
Grains and pixels are not treated as independent test units. Per-image point
scores are deterministic for these fixed labels; there is no invented per-image
confidence interval from resampling pixels.

The configuration audit checks identical ordered held-out IDs, manifest and
initial checkpoint hashes, recorded source, package versions, seed, epochs,
augmentation multiplier, batch size, inference options, and calibration.
Training sets are nested and disjoint from the test set. Hardware differs and
larger training sets receive more optimizer batches at the same epoch count.

## Measurement audit

The numeric calibration is 2.26 pixels per micrometre. The script tests the
released Jeffries functions against the current implementation on identical
normalized binary inputs for all ten images. It also reports sensitivity to
2.25 pixels per micrometre and raw versus processed GT. Matching formulas does
not establish identical original input preparation or independent physical
calibration. The current pipeline reads integer GT correctly using `labels > 0`,
matches exact image IDs, and retains failure cases instead of skipping them.

Use separate validation data for any normalization, probability/flow-threshold,
or loss changes. The inspected test images must not become a tuning set.
