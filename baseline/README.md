# Baseline suite

This folder contains implementations, pinned environments, configurations,
citations, and Unity job scripts for every requested comparison. None of these
methods uses the Voronoi consistency loss.

| Method / identifier | Role | Implementation |
| --- | --- | --- |
| Cellpose-SAM / `cellpose_sam` | Essential control for a future matched Voronoi intervention | Native full-network fine-tuning, cpsam weights, Cellpose 4.0.6 |
| Cellpose 2 / `cellpose2` | Established microscopy family | cyto2, Cellpose 2.3.2, native SGD fine-tuning |
| Cellpose 3 / `cellpose3` | Established microscopy family | cyto3 segmentation, Cellpose 3.1.1.1, native AdamW fine-tuning; no restoration stage |
| U-Net + watershed / `unet_watershed` | Supervised convolutional comparison | Four-level U-Net from scratch, BCE + Dice interior loss, distance watershed |
| SAM 2 / `sam2` | General foundation model | SAM 2.1 Hiera Small, fixed image-derived positive-prompt policy, inference only |
| StarDist / `stardist` | Shape-aware comparison | StarDist 2D from scratch, 32 rays, grid 2, TensorFlow |
| Classical watershed / `watershed` | Non-neural reference | Smoothing, Otsu threshold, distance peaks, marker-controlled watershed |

The additions are runnable research implementations, not independently validated
reproductions of each original paper's scores. GPU integration runs are pending.
The primary grain-paper reproduction remains available through `voron-baseline`
and `hpc/`; it uses a different protocol and must not be pooled with suite results.

## Papers

- **Cellpose-SAM:** Pachitariu, Rariden and Stringer (2025), [Cellpose-SAM: superhuman generalization for cellular segmentation](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1).
- **Cellpose 2.0:** Pachitariu and Stringer (2022), [Cellpose 2.0: how to train your own model](https://doi.org/10.1038/s41592-022-01663-4).
- **Cellpose 3:** Stringer and Pachitariu (2025), [Cellpose3: one-click image restoration for improved cellular segmentation](https://doi.org/10.1038/s41592-025-02595-5).
- **U-Net:** Ronneberger, Fischer and Brox (2015), [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
- **SAM 2:** Ravi et al. (2024), [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714). The implementation uses the authors' improved SAM 2.1 checkpoint, explicitly labeled in metadata.
- **StarDist:** Schmidt et al. (2018), [Cell Detection with Star-convex Polygons](https://arxiv.org/abs/1806.03535).
- **Watershed:** Vincent and Soille (1991), [Watersheds in digital spaces](https://doi.org/10.1109/34.87344). The concrete preprocessing and distance-marker pipeline here is our experimental configuration.

BibTeX entries are in [references.bib](references.bib). Upstream API sources:
[Cellpose](https://github.com/MouseLand/cellpose),
[SAM 2](https://github.com/facebookresearch/sam2), and
[StarDist](https://github.com/stardist/stardist).

## Environment setup on Unity

From the repository root in your Unity work directory, request a CPU allocation
and install methods serially. Multiple Cellpose releases and TensorFlow use
different dependencies and therefore have separate environments.

```bash
export PROJECT_ROOT="$PWD"
salloc --partition=cpu --cpus-per-task=4 --mem=16G --time=02:00:00
cd "$PROJECT_ROOT"
for method in cellpose_sam cellpose2 cellpose3 unet_watershed sam2 stardist watershed; do
    bash baseline/setup_unity.sh "$method"
done
exit
```

Add your Unity account flag when required. Setup uses `conda/latest`, creates
`.baseline-envs/METHOD`, installs the appropriate PyTorch/TensorFlow stack, checks
imports and package consistency, and fetches pretrained weights before GPU jobs.
Setup may need multiple CPU allocations depending on download speed. Each method
can be installed independently. No environments, datasets, or weights are tracked
in Git; allow substantial work-directory disk space for seven environments.

Cellpose 2/3 use PyTorch 2.2.2 + CUDA 12.1; Cellpose-SAM, U-Net and SAM 2 use PyTorch
2.6.0 + CUDA 12.4. StarDist uses TensorFlow 2.15.1 with its Linux CUDA dependencies,
and a CPU PyTorch install only to satisfy the common package dependency. StarDist
GPU availability is checked through TensorFlow, not `torch.cuda`. SAM 2's optional
custom CUDA build is disabled, avoiding a compiler/toolkit dependency. TensorFlow
and framework GPU visibility still need validation on Unity's allocated nodes.

## Prepare external inputs once

If the existing prepared manifest is unavailable, use the Cellpose-SAM environment
inside a CPU allocation:

```bash
.baseline-envs/cellpose_sam/bin/python -m voron_matsam.baseline download-data
.baseline-envs/cellpose_sam/bin/python -m voron_matsam.baseline prepare
```

Do not repeat downloading/preparation over an existing output directory. Dataset
location and exclusions are described briefly in the [root README](../README.md#external-data-and-weights).

## Freeze shared splits

```bash
.baseline-envs/watershed/bin/python -m baseline plan
```

The plan reserves five images from the original training pool for validation,
keeps its ten test micrographs fixed, and builds nested 2/5-shot subsets for five
seeds. All methods use the same saved plan and manifest hashes. Shots count full
annotated micrographs; physical specimen IDs are unavailable in the current
release, so this prevents micrograph/patch leakage without establishing independent
physical specimens. Use specimen-level partitions when those IDs become available.

## Smoke test each method before full jobs

```bash
SMOKE=1 bash baseline/submit.sh watershed
SMOKE=1 bash baseline/submit.sh unet_watershed
SMOKE=1 bash baseline/submit.sh cellpose_sam
SMOKE=1 bash baseline/submit.sh cellpose2
SMOKE=1 bash baseline/submit.sh cellpose3
SMOKE=1 bash baseline/submit.sh sam2
SMOKE=1 bash baseline/submit.sh stardist
```

These are explicitly submitted jobs, not a claim they have already run. A smoke
job uses two epochs / two requested steps for trainable methods and evaluates one
image; inspect its log and `run.json` status. It is excluded from final summaries.
Classical watershed runs on CPU. Other jobs request a suitable GPU. Resource and
time limits are starting estimates, not measured runtime guarantees.

After smoke jobs succeed, submit the full arrays:

```bash
for method in cellpose_sam cellpose2 cellpose3 unet_watershed sam2 stardist watershed; do
    bash baseline/submit.sh "$method"
done
```

Trainable-method arrays run both shot counts and five seeds, one job at a time
per method. SAM 2 and watershed default to a single job because they do not learn
from these subsets; their summary reports zero annotated training images and no
shot count. Submitting all methods allows seven independent arrays to proceed concurrently.
For a smaller first experiment, use `--array=0-1%1`; append an account or dependency
flag as needed. Retries require a new `RESULTS_ROOT`; completed output directories
are never overwritten. Checkpoint resume is not implemented.

## Individual runs, validation, and matched prompts

```bash
.baseline-envs/unet_watershed/bin/python -m baseline run \
  --method unet_watershed --shots 2 --seed 0 \
  --output outputs/baseline_suite/unet_watershed/2shot_seed0
```

Configurations live in `baseline/configs/`. Copy a configuration to an experiment
file and pass `--config PATH` to override it. Use `--evaluate-split validation` for
validation-only runs, allocate an equal tuning budget per method, and freeze the
chosen config before test evaluation. No defaults were selected using test results.
The runner passes training and validation labels to supervised adapters; final
test labels are used only by the shared evaluation code.

SAM 2 uses a deterministic Otsu/distance-watershed prompt generator and a maximum
of 512 positive prompts. No test GT centers, boxes or instance counts are used.
It saves XY coordinates under each run's `prompts/` directory. For exactly matched
prompts across SAM 2 comparisons, pass `--prompts-dir PATH_TO_SAVED_PROMPTS`;
the replay must be labeled `image_only` and obey the same prompt budget. This is
an automatic prompted-inference baseline, not SAM 2 fine-tuning or an oracle-prompt
experiment. Its detector's errors are part of end-to-end performance.

Add `--zero-shot` for an additional pretrained Cellpose 2/3/SAM comparison. Trainable
methods request 100 epochs and 30 updates per epoch with batch size 2. Native
Cellpose sampling/augmentation differs from U-Net/StarDist patch sampling; requested
budgets and framework metadata are recorded. Different algorithms need separately
validated learning rates and resource measurements. The current defaults provide
a controlled starting point, not proof of equal optimization quality.
Native Cellpose training resets its NumPy augmentation RNG by epoch; the selected
run seed still controls the few-shot subset and PyTorch RNG, but does not replace
those internal framework seed rules. This policy is recorded with training metadata.

Cellpose-SAM without Voronoi isolates the geometric intervention **only once** the
treatment uses the identical architecture, trainable parameters, supervision,
prompts, initialization, and optimization budget. That treatment trainer is still
future work. Keep a zero-loss control for any new seed/head architecture.

## Shared results

```bash
.baseline-envs/watershed/bin/python -m baseline summarize
```

Runs save `run.json`, source/manifest/plan/checkpoint hashes, `pip-freeze.txt`,
training artifacts, integer instance masks, `per_image.json` and `summary.json`.
The summary command excludes smoke and validation-only runs and rejects mixed
manifest/plan histories. Suite metrics are AP50, mean AP50:95, PQ, matched Dice,
boundary F1, false-split/merge rates, and signed count error.

AP here means `TP/(TP+FP+FN)` at fixed IoU thresholds, not COCO confidence-ranked
AP. Boundary F1 uses one-pixel Manhattan tolerance; false-split/merge overlap is
10% of both objects. These conventions are identical across suite methods but
differ from the original grain-paper reproduction pipeline. Report per-image
results and paired uncertainty across independent specimens when possible;
repeated seeds and grains from one micrograph are not independent specimens.
