# Baseline source audit — 2026-11-03

Scope: Mueez and Vyas, CVMI/CVPR Workshops 2026.
[Official proceedings](https://openaccess.thecvf.com/content/CVPR2026W/CVMI/html/Mueez_Bridging_Foundation_Models_and_ASTM_Metallurgical_Standards_for_Automated_Grain_CVPRW_2026_paper.html),
[full arXiv v1](https://arxiv.org/html/2604.18957v1),
[authors' implementation](https://github.com/mueez-overflow/ASTM-Grain-Size-Estimator/tree/80f305e9ae10d3f845914f6d773ecd2f8fcea651).

## Reported targets

[paper_targets.csv](../configs/paper_targets.csv) transcribes Cellpose-SAM rows
from Tables 1–2. The primary 60-grain, GT-superimposed, two-image result is G MAPE
1.88%. The headline 1.50% occurs at 90 grains with independent circles (Table 3).
These are reported results, not independently reproduced measurements.

Table 2's zero-shot density mean is 125.7, with mean internal/intercepted counts
38.4/17.7. Given its stated multiplier range 8.55–11.14 and shared circles,
the density mean must be at least `8.55 * (38.4 + 17.7/2) = 403.99`.
That row is inconsistent under the described protocol; request per-image results
or a correction. Do not replace its published values with guessed corrections.

## What the released files actually support

These findings come from reading the pinned checkout and inspecting the downloaded
archive locally, rather than assuming the README commands work:

| Finding | Consequence / implementation choice |
| --- | --- |
| `requirements.txt` is referenced but absent; `run.sh` is empty. | Supply reconstruction pins and cluster scripts here. |
| No original split CSVs, weights or per-image result tables are present. | Mark the seeded, sorted split as reconstructed. Numerical agreement cannot be promised. |
| Preparation expects matching image/mask stems. Archive stems are `RG10_1_1` versus `RGMask10_1_1`. | Pair by numeric micrograph ID and row/column; fail on incomplete grids. |
| Archive has 480 real image/mask pairs, arranged as 40 groups of 12, plus synthetic and pseudo-label folders. | Include only real `RG` / manual `RGMask` pairs. No synthetic pretraining. |
| Sample masks visibly have white interiors; release uses `>=128`, whereas the paper equation uses `<128`. | Default to bright interiors; record polarity, threshold, component filtering and erosion. |
| Released training calls Cellpose's optimizer on the whole network. | Full-network fine-tuning is the baseline. A frozen encoder with added heads is a later H1 design, not this reproduction. |
| README training flags do not match the actual training parser; prepared test directories in examples are not created by preparation. | Use an explicit manifest and runner rather than stringing together the example commands. |
| Jeffries script loads masks as 8-bit grayscale and thresholds at 127. Prepared GT files are integer instance TIFFs. | Read GT label IDs properly and use nonzero foreground; preserve raw masks for a separate sensitivity run. |
| Inference saves only rendered binary masks; count uses maximum label ID. | Save integer instances too and compact positive IDs before counting/evaluation. |
| Filename prefix matching in Jeffries can confuse IDs such as `RG1` and `RG10`. | Match exact manifest IDs, never first matching prefixes. |
| Training logs evaluation-set loss in released code. | Omit test data from training. Use a fixed final epoch with no test-based checkpoint choice. |
| Missing/empty masks can be skipped and zero-reference G entries filtered out. | Fail missing inputs; retain failure counts and return undefined MAPE where appropriate. |
| No U-Net, MatSAM or Qwen training/inference implementation is included. | This runnable reconstruction covers Cellpose-SAM. Rebuilding the other comparison models from incomplete settings would be a separate experiment. |

## Local data verification

The Kaggle primary source is [Artificial Grains and Real Grains](https://www.kaggle.com/datasets/peterwarren/voronoi-artificial-grains-gen),
version 4. Upstream code and checkpoint references are saved in
[`configs/upstream.json`](../configs/upstream.json). Dataset metadata, samples,
and generated audit outputs are not included in Git.
The primary archive SHA256 is
`514d213abb0ce7c13da6d8063256061aea763e1fe959e5251278dd86be3836c6`.
The metadata description's old image count and patch dimensions do not match the
actual v4 files. The inspected files have 400x400-pixel patches; reconstruction
therefore produces 1200x1600-pixel micrographs. No resizing is used to reconcile
this discrepancy.

Run `scripts/audit_prepared_data.py` to compare raw and processed GT statistics.
The initial reconstructed data's GT measurements do not exactly reproduce the
paper's reference means. This establishes a split/preprocessing ambiguity before
any network training, not a model failure. Keep this limitation with any measured
metric deltas. Do not optimize preprocessing against test-result agreement.

Circle selection, foreground rendering and scale must all be included when
requesting the original data-preparation protocol from the authors. The Kaggle
description reports 2.25 px/um while the paper's calibration is 2.26 px/um; both
values and their provenance are retained rather than silently conflated.

The checkpoint pin is Hugging Face revision
`7c61431b5fbb078f3296754bd15d9f51b320f837`, file `cpsam`, SHA256
`e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2`.
This is a reproducible starting checkpoint, not evidence of the exact revision
used by the authors. GPU smoke testing and quantitative replication are pending.
