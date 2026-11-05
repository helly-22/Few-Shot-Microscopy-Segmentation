# Reproduction checklist and execution record

Updated November 4, 2026. This follows the supplied checklist, excluding Qwen.
The Cellpose data/evaluation gate comes before the remaining trained models and
before Voronoi regularization. No published score is assigned to an extended
baseline that the reference paper did not benchmark.

## Current evidence

The original six Unity runs completed, but did not reproduce the fine-tuning
gains. Original 2-shot AP50 was 0.454843, versus the paper's 0.6457. Those runs
remain unchanged under `outputs/paper-20261103T234304Z`.

The checklist audit verified all 480 image/mask patch pairs and all 40 stitched
image/mask pairs pixel-for-pixel against the pinned archive. Bright mask pixels
are grain interiors; dark pixels track the annotated boundaries in the reviewed
eight-image contact sheet. We retain `>=128`, rather than blindly reversing it
to match the contradictory paper equation.

The legacy masks were uint32 and contained 612 IDs disconnected under
4-connectivity, including 119 disconnected under 8-connectivity. The validated
variant erodes first, labels with 4-connectivity, removes each component below
200 pixels, and writes uint16 TIFFs. All 40 pass connectivity, dtype and minimum
area checks. This changes the total instance count from 11,174 to 11,191. It is a
documented alternative to the released label-before-erosion implementation;
it is not a claim to have recovered the authors' unavailable original masks.

Artifacts are under `outputs/checklist/data_gate/`: `summary.json`,
`per_image_data_audit.json`, `split_policy.json`, `polarity_review.png`, and
`legacy_instance_vs_binary.json`.

The model-independent evaluator in `src/voron_matsam/paper_metrics.py` reproduces
all stored 2-shot per-image AP50, mAP, boundary F1 and count scores within 1e-7.
It uses outer boundaries, disk(2) dilation and disk(3) tolerance. The existing H1
and `baseline_suite_v1` boundary scores remain separate definitions. New paper
comparisons must use the common paper evaluator.

The binary export audit records the original instance count and connected
component counts after rendering, separately for each of the ten test images.
Across the ten images, 1,990 original predictions become 2,001 8-connected
components in the binary exports. Binary exports can change object counts; segmentation counts always come from
the original integer masks. Jeffries uses the explicitly recorded binary
representation, with its 8-connected counting rule.

## Frozen controlled protocol

- Dataset: `data/processed/connected-v2/manifest.json`; source archive checksum
  remains pinned in `baseline_data.py`. Manifest and run files record file hashes.
- 2-shot training: RG18, RG12. Nested sizes remain 2, 4, 10, 20 and 30.
- Validation: RG24, RG9, RG4, RG11, RG1, taken from the training pool and excluded
  from the 2-, 4-, 10- and 20-shot subsets.
- Test: RG8, RG34, RG14, RG15, RG17, RG22, RG23, RG25, RG10, RG16.
- The 30-shot final refit may use all 30 only after settings are frozen using
  smaller fits. It cannot simultaneously use an independent five-image
  micrograph validation set from those same 30. Do not describe it that way.
- Calibration: 2.26 pixels/um, the released code's numeric value. Physical scale
  and physical specimen independence remain unverified.
- Primary grain-size endpoint: superimposed Jeffries circles, target 60 internal
  grains, half weight for intercepted grains, processed GT representation.
  Record target failures and undefined measurements; never drop them silently.
- Test images have already been inspected. This is a controlled diagnostic and
  reproduction benchmark, not an untouched external generalization test.

The released filename sort order (`RG1_stitched.png`, etc.) produces a different
seed-42 split from bare-ID sorting. Its alternative first pair is RG19/RG13,
and its test set is RG8, RG35, RG15, RG16, RG18, RG23, RG24, RG26, RG11, RG17.
Both are recorded in `split_policy.json`. We keep the existing IDs to isolate
the mask change. Neither split is author-confirmed; matching a published score
must not be used to choose between them.

## Ordered checklist

| # | Work | Status and acceptance condition |
|---|---|---|
| 1 | Verify 480 pairs and reconstruct 40 | Complete; source pixels verified |
| 2 | Inspect polarity and regenerate connected uint16 masks | Complete; separate version, zero fragmented IDs |
| 3 | Freeze memberships, hashes and calibration | Complete for controlled protocol; author split/physical scale unknown |
| 4 | Unit-test AP, boundary F1, counts and Jeffries | Complete; 62 local tests, 61 in frozen Unity snapshot |
| 5 | Cellpose zero-shot reference | Complete on validated labels: AP50 0.405532 vs 0.4006; raw outputs saved |
| 6 | Fresh 2-shot training, foreground and flow inspection | Both precision fits and galleries complete; manual gallery review pending |
| 7 | Validation-only reconstruction/normalization ablations | Complete for both fits and the inference-precision control; test settings unchanged |
| 8 | Independent 4/10/20/30 fits | Original protocol complete; validated protocol waits for Cellpose gate |
| 9 | Four-stage U-Net, BCE+Dice, 30 images | Complete; AP50 0.5354, documented reconstruction |
| 10 | Official MatSAM, SAM ViT-L zero-shot | Complete; AP50 0.5192; recipe deviations recorded, separate from SAM2 |
| 11 | Connected-component classical baseline | Complete; AP50 0.2750 on common protocol |
| 12 | Marker-controlled watershed | Complete; AP50 0.1774 on common protocol |
| 13 | U-Net + watershed | Complete; AP50 0.5858, exact same checkpoint as plain U-Net |
| 14 | StarDist 2D | Complete; AP50 0.6481 after 25-image development and 30-image final refit |
| 15 | Evaluate every method through common evaluator | All six alternatives and the Cellpose reference use the common evaluator |
| 16 | Superimposed Jeffries target 60 as primary | Implemented; preserve measurement/representation audit |
| 17 | Per-image galleries and uncertainty | Cellpose and five alternatives generated; representative figures checked, exhaustive review pending |
| 18 | Additional independent training seeds | Pending after protocol settles |
| 19 | Add Voronoi regularization | Deferred until all prerequisite controls are reviewed |

## Unity diagnostic execution

Unity run root: `/work/pi_yzick_umass_edu/helly/Few-Shot-Microscopy-Segmentation/outputs/checklist-20261104T1715Z`.
Preparation job 66697322 and diagnostic job 66697323 completed successfully.
The source-only bundle is frozen in that run's `source/` directory;
the original project source directory was not replaced.

Validated-label zero-shot results (all ten images): AP50 0.4055321,
mAP50-95 0.2112530, boundary F1 0.5026762, signed count error -122.2.
The corresponding paper values are 0.4006, 0.2089, 0.5022 and -125.0.
This is close numerical agreement for the zero-shot reference, with the split
and preprocessing limitations above. Local summary:
`outputs/checklist/zero_shot_validated_summary.json`.

The validated-mask 2-shot fit completed on an NVIDIA A100-PCIE-40GB with AP50
0.4542076, mAP50-95 0.2341547, boundary F1 0.5571455, and signed count error -96.2.
The corresponding paper values are 0.6457, 0.3691, 0.7704 and -17.6. The result
remains near the old AP50 of 0.4548429; the mask correction alone does not resolve
the fine-tuning gap. The old run also used a different GPU model, so this is not
an exact hardware-matched mask-only intervention. Precision and validation
reconstruction controls remain necessary. Local JSONs: `outputs/checklist/05_seed0/`.

The completed float32 2-shot control reached AP50 **0.6565442** and primary
superimposed-60 G MAPE **1.12351%**, compared with bfloat16 AP50 0.4542076 and
G MAPE 7.94231%. Both new fits ran on NVIDIA A100-PCIE-40GB GPUs with matching
recorded training settings apart from precision. The paper reports AP50 0.6457
and G MAPE 1.88%. These numbers do not establish exact paper reproduction:
original author splits, physical calibration and some preprocessing remain
unconfirmed, and this control changes the parameter-precision policy. Full
per-image results, intervals and validation controls are in the generated report.

`hpc/checklist_prepare.sbatch` builds the separate validated dataset.
`hpc/checklist_cellpose.sbatch` runs tests, zero-shot, a fresh 2-shot fit from the
pinned original cpsam weights, and validation diagnostics. Both test inference
runs save instance labels, binary exports, foreground scores and flow fields.
Training remains 100 epochs, 30 copies per original, batch 2, LR 1e-6, weight
decay 0.2, seed 0, and Cellpose 4.0.6's default bfloat16 parameter dtype. Precision
is recorded, not silently changed alongside the mask experiment.

The validation diagnostic uses 13 configurations: the reference, four additional
foreground thresholds, three additional flow thresholds, three additional
minimum sizes, and two normalization alternatives. These are separate
one-factor-at-a-time sweeps, not a factorial search. Reference normalization is
Cellpose's default; the explicit percentile alternative is 0.5/99.5; the third
mode disables normalization. These are diagnostic choices, not paper settings.

Raw reference flows are reused for reconstruction sweeps. For all three
normalization modes on each validation image, cached default reconstruction
must exactly match native Cellpose instance output before proceeding. Threshold
selection uses only the five validation images. Bootstrap intervals resample
whole images 20,000 times with seed 20261104 and do not measure training-seed
variation. No selected setting is automatically applied to the test set yet.

The additional CPU checkpoint audit (job 66697502, completed) compared the old
2-shot checkpoint with the pinned initial weights after casting each initial
tensor to its saved dtype. Of 304,628,928 compared floating-point values,
3,808,895 changed (1.25034%); 57 of 348 compared tensors were exactly unchanged.
The diameter metadata keys differ in shape and are listed as unmatched, rather
than included in those totals. This is descriptive evidence, not proof that
bfloat16 caused the accuracy gap.

`hpc/checklist_precision.sbatch` follows the main diagnostic with a matched
float32 policy control (job 66697720, dependent on 66697323). It first evaluates the same bfloat16-trained checkpoint
in float32 on validation, isolating inference precision. It then independently
trains from the original cpsam checkpoint in float32 with the same seed, images,
augmentation and optimizer settings, and evaluates the new checkpoint in
float32. This latter comparison changes the complete parameter-precision policy,
including initial rounding and inference; it is not a pure training-only dtype
intervention. Settings are still selected on validation, not the ten test images.

Local tests: 62 passed. The frozen main Unity bundle contains the 61-test version;
the added local regression test verifies that checkpoint dtype rounding is not
counted as a learned update. All 40 local and Unity uint16 mask-file hashes match;
PNG-file hashes differ, while decoded pixels in all ten archived Unity test
images and their raw masks match the local reconstruction exactly.

Report job 66697805 depends on the precision control. It builds
`report/report.html`, `report/results.json`, `report/per_image_scores.csv`,
all ten four-panel and raw-output figures for both 2-shot fits, and the portable
`checklist_results.zip`. It reports paired differences and whole-image
uncertainty, with manual gallery review explicitly pending. A synthetic
end-to-end check generated all 40 expected figures and recovered a known paired
AP difference; representative rendered figures were inspected.

Execution update: diagnostic 66697323 and precision control 66697720 completed
with exit 0:0. The original report job failed before rendering because Matplotlib
was absent from the baseline environment. Installed `matplotlib==3.9.4` from the
new `requirements/report.txt` pin; report retry 66698665 completed successfully. Existing
NumPy, SciPy, PyTorch, Cellpose, Albumentations and Pillow versions were unchanged.
For a fresh report environment, run
`.hpc-env/bin/python -m pip install -r requirements/report.txt` before reporting.

Sync verification: key experiment source and report hashes match between local
and Unity's run snapshot. This is not a full working-tree mirror: newer local
documentation and the local-only watcher are separate. The GitHub update contains
code, tests, job scripts and documentation; datasets, weights and generated
outputs remain excluded.

The local `scripts/watch_unity_checklist.py` process monitors only these recorded
jobs for up to 24 hours. It submits no jobs. After successful report completion,
it downloads the ZIP, checks its SHA-256 and validates extraction paths, then
saves the HTML under
`outputs/checklist_results/checklist-20261104T1715Z/report/report.html`.
The report was downloaded and its SHA-256 verified successfully. The watcher
stops on job failure or repeated connection errors. State and logs are in
`outputs/runtime/unity-checklist.json` and `outputs/runtime/checklist-watch.log`.

To regenerate local data checks:

```powershell
.\.venv\Scripts\python.exe -m voron_matsam.baseline prepare --output data/processed/connected-v2 --label-policy connected_uint16
.\.venv\Scripts\python.exe scripts/checklist_data_gate.py
.\.venv\Scripts\python.exe -m pytest -q
```

Preparation refuses a nonempty destination; use the existing verified manifest
instead of overwriting it. Dataset files, checkpoints and outputs stay excluded
from Git.

## Sources

- [Mueez and Vyas reference paper](https://arxiv.org/html/2604.18957v1).
- [Released project code](https://github.com/mueez-overflow/ASTM-Grain-Size-Estimator),
  audited commit `80f305e9ae10d3f845914f6d773ecd2f8fcea651` (see `docs/PAPER_AUDIT.md`).
- [Cellpose 4.0.6 model implementation](https://github.com/MouseLand/cellpose/blob/v4.0.6/cellpose/models.py)
  and [training implementation](https://github.com/MouseLand/cellpose/blob/v4.0.6/cellpose/train.py).
- [Official MatSAM](https://github.com/USTB-AI3DVIP/MatSAM).
