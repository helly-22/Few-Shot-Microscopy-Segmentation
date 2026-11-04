# Run the baseline on UMass Unity

The prepared workflow runs **Cellpose-SAM zero-shot and all five fine-tuned
fractions**. It records comparisons against the paper rather than assuming that
matching results have already been obtained. Read [the source audit](PAPER_AUDIT.md)
before interpreting differences.

## Clone on Unity

Clone the code into your allocated work directory using a Unity terminal. Your
research group's `/work/...` path and account are specific to your Unity access.
Datasets, weights, environments, and generated outputs are not tracked in Git.

Inside a Unity terminal, from your selected work directory:

```bash
git clone https://github.com/helly-22/Few-Shot-Microscopy-Segmentation.git
cd Few-Shot-Microscopy-Segmentation
export PROJECT_ROOT="$PWD"
mkdir -p logs
```

Do not copy a Windows `.venv` to Unity. If Git is unavailable, generate a
source-only transfer archive with `python scripts/build_unity_bundle.py` locally.

## Install and prepare

To queue the complete workflow from the login node, run:

```bash
bash hpc/submit_reproduction.sh --account=YOUR_UNITY_ACCOUNT
```

This submits a CPU preparation job, a GPU smoke test, the six-run baseline array
(one GPU at a time), and a CPU comparison job. Each stage requires successful
completion of the previous stage. Preparation installs the environment, runs the
tests, downloads and verifies the data and checkpoint, and audits the prepared
data. The launcher prints job IDs and saves them in a new
`outputs/paper-<UTC timestamp>/jobs.tsv`. Results and the final
`paper_comparison.json` are saved under that same run directory. Use `squeue --me`
to check progress and inspect the corresponding files in `logs/` for failures.
Submission alone does not establish that training or reproduction succeeded.

For manual control over each stage, follow the steps below.

Request an interactive CPU allocation for installation and image preparation:

```bash
salloc --partition=cpu --cpus-per-task=4 --mem=16G --time=01:00:00
cd "$PROJECT_ROOT"
bash hpc/setup_unity.sh
bash hpc/fetch_assets.sh
.hpc-env/bin/python scripts/audit_prepared_data.py
exit
```

If your group requires an account flag, add `--account=YOUR_UNITY_ACCOUNT` to
`salloc` and `sbatch`. Unity documents `module load conda/latest` and work-directory
environments in its [Conda guide](https://docs.unity.rc.umass.edu/documentation/software/conda/).

Setup pins Python 3.11, PyTorch 2.6.0 / torchvision 0.21.0 with CUDA 12.4 wheels,
Cellpose 4.0.6 and Albumentations 1.3.1. These are reconstruction choices, since the
authors did not supply a lockfile. All resolved dependencies are saved after
installation and inside every run. The matching PyTorch installation is listed
in [PyTorch's previous versions](https://pytorch.org/get-started/previous-versions/).

`fetch_assets.sh` downloads the pinned 1.23 GB cpsam checkpoint and verifies its
SHA256. It separately downloads the external grain archive when absent.
Data preparation takes only `RG` and `RGMask`, checks all
40 complete 3x4 grids, converts masks and creates a hash-checked manifest. Synthetic
grains and algorithm-generated pseudo-label directories are excluded.

The default setup environment is `.hpc-env` under the project. Override its path
consistently using `export ENV_PREFIX=/your/work/path/grain-env` before installation
and submission if needed. The scripts keep caches under the project as well.
Allow roughly 20 GB of disk for initial setup and more for results/checkpoints.
The largest split materializes its augmented data and flows in memory; the batch
template requests 64 GB host RAM as an initial allocation to validate.

## Run a GPU smoke test first

Back on the login node, in the same project directory:

```bash
export PROJECT_ROOT="$PWD"
mkdir -p logs
sbatch --export=ALL hpc/smoke.sbatch
squeue --me
```

This trains two images for two epochs with one copy per image, then evaluates one
test image. Its results are marked `smoke_test_only` and excluded from paper
comparisons. Check `logs/smoke-JOBID.out`, `logs/smoke-JOBID.err` and the final
`outputs/smoke-JOBID/run.json` status before launching the full sweep.

GPU templates use the `gpu-highvram` partition and constrain devices to A40, A100,
L40S or H100: CUDA-compatible, BF16-capable choices with substantial VRAM. Unity's
[GPU guide](https://docs.unity.rc.umass.edu/documentation/tools/gpus/) explains GPU
requests, and its [feature list](https://docs.unity.rc.umass.edu/documentation/cluster_specs/features/)
documents the model names. These requests are starting resource estimates; wall
time and memory still need validation on your allocation. The environment targets
x86 Linux CUDA GPUs; Grace Hopper and Blackwell require a separately validated
software build. Prefer this nonpreempting partition for these jobs: automatic resume
after preemption is not implemented.

## Run the paper baseline sweep

After a successful smoke test:

```bash
bash hpc/submit_baselines.sh
```

The array maps tasks 0–5 to `zero`, `05`, `10`, `25`, `50`, `75`. The fractions use
2, 4, 10, 20 and 30 full micrographs. Ten micrographs remain fixed for evaluation.
The array runs one GPU job at a time by default. Examples of optional overrides:

```bash
bash hpc/submit_baselines.sh --account=YOUR_UNITY_ACCOUNT
# Only zero-shot and the smallest fine-tuning split:
bash hpc/submit_baselines.sh --array=0-1%1
# Another training/augmentation seed, preserving the same data subsets:
SEED=1 bash hpc/submit_baselines.sh
```

Each job starts from the pinned checkpoint and uses the native flow/probability
loss, full-network fine-tuning, 100 epochs, batch size 2 and 30 copies per training
micrograph including the original. The Voronoi code is not part of these runs.
Test images are not passed into training or used to select checkpoints. The final
saved checkpoint is reloaded for evaluation. Ten-epoch saves provide recovery
artifacts, but the runner currently restarts a failed job from the initial model;
it does not implement optimizer-state resume.

Each output directory must be new. To rerun a failed attempt, use a new results
root instead of mixing it with old files:

```bash
RESULTS_ROOT="$PWD/outputs/baselines-retry" bash hpc/submit_baselines.sh --array=1
```

## Inspect and compare

```bash
.hpc-env/bin/python -m voron_matsam.baseline compare \
  --results outputs/baselines --output outputs/paper_comparison.json
```

Each completed run saves `run.json`, `pip-freeze.txt`, final weights and losses
for fine-tuning, original integer instance predictions, rendered binary masks,
per-image segmentation metrics, per-image Jeffries measurements and `summary.json`.
Both 10–100 target sweeps and both circle modes are evaluated. Comparison uses the
primary 60-grain, superimposed targets and reports deltas, not automatic success.
For the headline result, inspect the independent-circle, target-90 summary entry.

Jeffries GT defaults to the processed-instance foreground. Raw-mask foreground is
an explicitly different diagnostic (`--jeffries-gt raw`). It does not silently
threshold integer instance TIFF values at 128. Calibration is 2.26 px/um by default
and is saved in each run. The archive's description states 2.25 px/um; check scale
bar provenance before treating the absolute grain density as validated. Ignored
regions and physical specimen IDs are not provided by this release, so micrograph
grouping prevents patch leakage but cannot establish specimen-independent splits.

Actual GPU runs, runtime/VRAM validation and experimental reproduction remain to
be performed on Unity. No remote jobs were submitted from the local setup.
