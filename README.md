# Few-Shot Microscopy Segmentation

Research code for reproducing a Cellpose-SAM microscopy baseline and investigating
soft Voronoi constraints for few-shot instance segmentation of metallic grains.

The baseline runs the native Cellpose-SAM model with zero-shot inference or
full-network fine-tuning. The geometry module provides a differentiable weighted
Voronoi layer for a subsequent controlled experiment; it is **not yet integrated
into the baseline trainer**.

**Status:** local unit and orchestration tests pass. Actual Cellpose-SAM GPU
training, Unity smoke testing, and replication of published results are pending.
The runner integration test uses a mock model.

## Comparison baselines

The **[baseline suite](baseline/README.md)** contains separate adapters, pinned
environments, configurations, shared splits/evaluation, Unity jobs, and
[paper citations](baseline/references.bib) for:

- Cellpose-SAM without Voronoi (essential geometry control).
- Cellpose 2.0 and Cellpose 3.
- U-Net followed by watershed instance separation.
- SAM 2.1 with repeatable image-derived prompts.
- StarDist 2D.
- Classical marker-controlled watershed.

Use `python -m baseline --help` and the suite guide to run these comparisons.
The original paper reconstruction below remains a separate protocol; suite
results must not be compared against its published numbers as if metrics matched.

## Repository structure

```text
configs/                 Published benchmark targets and upstream version references
baseline/                Multi-method suite, environments, configurations, and citations
docs/                    Unity guide, source audit, and H1 experimental protocol
hpc/                     Unity environment setup, asset fetching, and Slurm jobs
requirements/            Pinned baseline dependencies
scripts/                 Data audit and source-only transfer bundle utilities
src/voron_matsam/
    baseline.py          Training, inference, evaluation, and comparison CLI
    baseline_data.py     External input preparation and manifest validation
    geometry.py          Differentiable weighted Voronoi assignments and loss
    jeffries.py          Circular grain-count and grain-size evaluation
    metrics.py           Instance and boundary metrics for H1
    splits.py            Paired few-shot splits with specimen holdouts
tests/                   Geometry, data-handling, metrics, and runner tests
```

## Local installation and tests

Use Python 3.11 for the pinned baseline environment. On Linux:

```bash
git clone https://github.com/helly-22/Few-Shot-Microscopy-Segmentation.git
cd Few-Shot-Microscopy-Segmentation
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU wheels for local correctness tests:
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/baseline.txt
python -m pip install --no-deps -e .
python -m pytest -q
```

In Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
CPU installation is for correctness checks; use the Unity environment below for
GPU experiments. Dependency pins are reconstruction choices because the authors'
exact environment was not released.

## External data and weights

Data and model weights are **not included in this repository**. The baseline uses
the real `RG` / `RGMask` pairs from [Warren's Kaggle dataset, version 4](https://www.kaggle.com/datasets/peterwarren/voronoi-artificial-grains-gen).
The asset script downloads these inputs and the pinned cpsam weights into ignored
`data/` directories; do not commit inputs, masks, checkpoints, or run outputs.

## Run on UMass Unity

Clone the repository into your allocated Unity work directory. From the repository
root, request a CPU allocation to install dependencies and prepare inputs:

```bash
export PROJECT_ROOT="$PWD"
salloc --partition=cpu --cpus-per-task=4 --mem=16G --time=01:00:00
cd "$PROJECT_ROOT"
bash hpc/setup_unity.sh
bash hpc/fetch_assets.sh
exit
```

Back in the project directory on the login node, submit the short GPU smoke job:

```bash
export PROJECT_ROOT="$PWD"
mkdir -p logs
sbatch --export=ALL hpc/smoke.sbatch
squeue --me
```

Inspect `logs/smoke-JOBID.out`, `logs/smoke-JOBID.err`, and the completed status in
`outputs/smoke-JOBID/run.json`. After that job succeeds, submit the baseline array:

```bash
bash hpc/submit_baselines.sh
```

The array runs zero-shot inference and fine-tuning with 2, 4, 10, 20, and 30 images,
using the same ten held-out micrographs. It requests one GPU at a time. Fine-tuning
uses 100 epochs, batch size 2, learning rate `1e-6`, weight decay `0.2`, and 30 copies
per training image, including the original. The final checkpoint is evaluated;
test images are not supplied to the optimizer or used for checkpoint selection.

Add your group account when required, for example
`bash hpc/submit_baselines.sh --account=YOUR_UNITY_ACCOUNT`. For allocation details,
alternate seeds, retry behavior, and resource limits, see the
[Unity run guide](docs/UNITY_BASELINE.md).

## Run an individual baseline

Inside a GPU allocation, using the environment installed by the Unity setup:

```bash
.hpc-env/bin/python -m voron_matsam.baseline doctor
.hpc-env/bin/python -m voron_matsam.baseline run \
  --manifest data/processed/paper-bright/manifest.json \
  --checkpoint data/models/cpsam \
  --split 05 --seed 0 \
  --output outputs/baselines/05_seed0
```

Use `--split zero` for zero-shot inference, or `05`, `10`, `25`, `50`, `75` for
fine-tuning. Output directories must be new. `--help` lists the remaining options.
Runs save environment and source hashes, integer instance predictions, per-image
metrics, and Jeffries evaluations for both circle-placement modes.

## Compare with the paper

```bash
.hpc-env/bin/python -m voron_matsam.baseline compare \
  --results outputs/baselines \
  --output outputs/paper_comparison.json
```

This reports differences from [published targets](configs/paper_targets.csv).
It does not declare replication success. Original author split manifests and
environment pins are unavailable, and the source audit identifies preprocessing
ambiguities and an inconsistent reported row. Read the
[paper/code audit](docs/PAPER_AUDIT.md) before interpreting numerical comparisons.

## Voronoi experiment

The standalone geometry helpers and metrics support H1: whether a soft geometric
constraint reduces false splits and improves boundaries without increasing false
merges. See the [H1 protocol](docs/H1_EXPERIMENT.md) for matched controls, split
requirements, and evaluation conventions. Integrating the loss into a trainable
Cellpose adapter remains future work.

## Reference

Abdul Mueez and Shruti Vyas. *Bridging Foundation Models and ASTM Metallurgical
Standards for Automated Grain Size Estimation from Microscopy Images.* CVMI,
CVPR Workshops, 2026. [Paper](https://arxiv.org/abs/2604.18957) ·
[Authors' implementation](https://github.com/mueez-overflow/ASTM-Grain-Size-Estimator).

This repository is an independent baseline reconstruction and research extension.
