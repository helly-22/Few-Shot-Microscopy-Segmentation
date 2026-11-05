# Other models on the validated grain benchmark

The `baseline.benchmark` runner is separate from `baseline_suite_v1`. It uses the
connected uint16 ground truth, the fixed ten test images, the common paper AP
and boundary evaluator, and superimposed Jeffries circles with target 60. No
Voronoi loss is used. These are documented reconstructions and extensions;
author splits, physical scale, and some published inference settings remain unknown.

## Results available November 4, 2026

| Model | AP50 | 95% image-bootstrap CI | mAP50-95 | Boundary F1 | Signed count error | G MAPE |
|---|---:|---|---:|---:|---:|---:|
| Connected components | 0.2750 | 0.2515-0.2969 | 0.1056 | 0.5179 | -63.4 | 15.38% |
| Classical watershed | 0.1774 | 0.1409-0.2121 | 0.0732 | 0.5414 | +269.3 | 15.39% |
| MatSAM / SAM ViT-L reconstruction | 0.5192 | 0.4933-0.5457 | 0.2700 | 0.7095 | +39.9 | 3.38% |
| U-Net + morphology, 30-image refit | 0.5354 | 0.5039-0.5675 | 0.2937 | 0.7750 | -67.6 | 4.85% |
| Same U-Net + watershed | 0.5858 | 0.5635-0.6077 | 0.3066 | 0.8107 | +29.8 | 1.87% |
| StarDist, 30-image refit | 0.6481 | 0.6269-0.6669 | 0.3640 | 0.7292 | -36.0 | 2.20% |
| Cellpose-SAM 2-shot float32 reference | 0.6565 | 0.6329-0.6784 | 0.3758 | 0.7814 | -8.2 | 1.12% |

These intervals resample ten whole images, 20,000 times with seed 20261104.
They do not measure training-seed or independent-specimen variability. The
classical methods used five validation annotations for parameter selection;
MatSAM used fixed settings and no annotations for selection. Cellpose used two
training images with the reference test-inference settings. These annotation
budgets differ and must accompany comparisons.

The paper reports MatSAM AP50 0.2688, mAP 0.1224, boundary F1 0.6559, count error
+151.6 and G MAPE 10.61%. Our MatSAM recipe does not recover all hidden settings,
so its higher score is not a confirmed reproduction or a proven improvement over
the authors' experiment. Classical methods and StarDist have no targets in that
paper. U-Net's published 30-image AP50 target is 0.4717, but its hidden settings
also require documented approximation.

## Methods and selection

- **Connected components:** image-only percentile normalization, grayscale mean,
  Gaussian smoothing and Otsu threshold. Select among five predefined variants
  on validation AP50; label four-connected regions and remove areas below 200.
- **Watershed:** same image-only preprocessing, distance-transform markers and
  watershed. Seven predefined validation candidates; no test tuning. The current
  result has substantial over-segmentation (+269.3 instances per image on average).
- **U-Net and U-Net + watershed:** one four-stage network with widths
  32/64/128/256, BCE+Dice, AdamW LR 1e-4, weight decay 1e-4, batch 2, 256-pixel
  patches, 100 epochs and 100 steps per epoch. Train a development model on 25
  images; select postprocessing on the other five. Reinitialize with the same
  seed and refit on all 30 after settings freeze. Both test variants use the
  exact same final checkpoint and cached probability maps. Four stages, widths,
  optimizer and morphology are explicit reconstruction choices, not recovered
  author settings.
- **StarDist 2D:** 32 rays, grid 2, base 32, 256-pixel patches, LR 0.0003, batch 4,
  400 epochs, 100 steps per epoch. Development on 25 images and selection on five,
  then final refit on 30 with settings frozen. The upstream final-refit API
  requires monitoring data; a training image is used only for loss logging,
  with constant LR, no early stopping and final-epoch checkpoint reload. This
  monitor is not independent validation. Thresholds are never selected from it.
- **MatSAM:** use the official pinned prompt and mask implementation with
  SAM ViT-L, fixed Canny ROI centroids plus a 32-by-32 grid, IoU threshold 0.90,
  stability 0.92, NMS 0.80 and no crop layers. The unchanged released prompt
  class retains its non-square XY normalization convention. The optional Gala
  metrics import is avoided by loading only the unchanged classes needed for
  prompting. Batch 32 replaces the notebook's 256 for memory. Score-ordered
  instance export retains regions of 200-50,000 pixels and suppresses overlaps;
  the upstream example instead produces a binary union. Those differences,
  ViT-L versus the notebook's ViT-H default, and missing grain-paper settings
  are recorded in the run protocol. This is SAM/MatSAM, not SAM2.

All methods save integer predictions, rendered binary masks, per-image AP,
boundary F1, counts, PQ, matched Dice, foreground Dice, split/merge rates and
Jeffries measurements. Matched Dice excludes unmatched objects; interpret it
with PQ and count error. Classical, MatSAM and both U-Net results completed on
Unity. StarDist also completed its 400-epoch development fit and 400-epoch
30-image final refit. All six alternatives and the Cellpose reference are now
available on the shared ten-image test set.

Cellpose minus StarDist paired AP50 difference is 0.00848, with a whole-image
bootstrap 95% interval of -0.00736 to 0.02350 (20,000 draws, seed 20261104).
The paired t interval is -0.01042 to 0.02738. This single-seed comparison does
not establish a reliable ranking between them; their annotation budgets also differ.

The two U-Net outputs share final checkpoint SHA-256
`2fb7b088b05aee93f21b8989a1b7bbf038cd66e772985f34a7d6f595cc96a0c6`.
Validation selected foreground threshold 0.6 with one-pixel erosion for morphology,
and threshold 0.4 with marker minimum distance 16 for watershed. Thus the learned
features are held fixed while each postprocessing policy is calibrated separately;
this does not isolate the watershed algorithm from its threshold settings.

## Run

Use an allocated CPU node for environment preparation and a GPU allocation for
training/inference. Existing baseline dependencies cover classical methods,
U-Net and MatSAM. Use `requirements/report.txt` for reports. StarDist needs the
separate environment from `baseline/setup_unity.sh stardist`.

```bash
# In the repository, using the existing prepared manifest:
.hpc-env/bin/python -m baseline.benchmark --method connected_components \
  --manifest data/processed/connected-v2/manifest.json --output outputs/cc-run
.hpc-env/bin/python -m baseline.benchmark --method watershed \
  --manifest data/processed/connected-v2/manifest.json --output outputs/ws-run

# In a CPU allocation; weights and upstream code remain ignored by Git:
.hpc-env/bin/python -m baseline.fetch_matsam

# In GPU allocations:
.hpc-env/bin/python -m baseline.benchmark --method unet \
  --manifest data/processed/connected-v2/manifest.json --output outputs/unet-run
.baseline-envs/stardist/bin/python -m baseline.benchmark --method stardist \
  --manifest data/processed/connected-v2/manifest.json --output outputs/stardist-run \
  --epochs 400 --steps 100
.hpc-env/bin/python -m baseline.benchmark --method matsam \
  --manifest data/processed/connected-v2/manifest.json --output outputs/matsam-run \
  --checkpoint data/models/matsam_sam_vit_l.pth --upstream external/matsam-benchmark
```

Each output directory must be new. For snapshot deployment, run from the
snapshot directory and pass absolute paths to the data and output directories;
otherwise Python may import an older `baseline` package from the working directory.
`hpc/benchmark.sbatch` implements that rule. StarDist's CUDA libraries from its
isolated environment are added to `LD_LIBRARY_PATH` in that script.

Current run root: `outputs/other-models-20261104` on Unity. Completed per-image
JSONs have been downloaded under local `outputs/other_models/`. The report
builder `scripts/report_other_models.py` produces a common comparison, per-image
CSV and ten-image galleries for every completed method. The complete report
finished successfully after both neural jobs and was downloaded with checksum verification.
All datasets, weights, predictions and generated reports remain excluded from Git.

Local final HTML report:
`outputs/other_models/final/other-models-20261104/report/report.html`.
It contains all six completed alternatives, the Cellpose reference, per-image
CSVs, bootstrap intervals and 60 four-panel figures. Representative MatSAM,
U-Net and watershed figures were visually checked; exhaustive manual review is
still pending. Job IDs: classical 66701648, U-Net 66701958, MatSAM 66701964
(completed), StarDist 66701965 and final report 66702548 (both completed, exit 0:0).

The configurable watcher monitors only recorded jobs and downloads the complete,
checksummed report after completion:

```powershell
.\.venv\Scripts\python.exe scripts/watch_unity_checklist.py --metadata outputs/runtime/unity-other-models.json
```

Its state is saved in that runtime JSON. It stops on failures, repeated SSH
errors or its 24-hour limit; it never submits additional jobs.

## References

- [Reference grain paper](https://arxiv.org/html/2604.18957v1).
- [Official MatSAM](https://github.com/USTB-AI3DVIP/matsam), pinned commit
  `cbea7edaada991d88d7dfee656bd7e3dac09863f`.
- [Official SAM checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
- [StarDist 0.9.1 implementation](https://github.com/stardist/stardist/tree/0.9.1).
