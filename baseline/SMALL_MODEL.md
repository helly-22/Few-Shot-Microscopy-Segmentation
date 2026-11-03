# First low-compute segmentation run

Use the small U-Net + watershed profile before installing or launching the full
suite. It has **485,817 trainable parameters**, compared with 7,760,097 in the
default U-Net configuration. It produces instance masks and needs no pretrained
model weights. Qwen vision-language models remain a separate counting experiment.

The profile uses base width 8, 128x128 training patches, batch size 1, FP32,
two annotated training micrographs, and 100 optimizer updates (10 epochs x 10
steps). It evaluates the fixed five-image validation set. This short budget is
an initial learning/compute check, not a completed few-shot benchmark.

## On Unity

From the repository root in your work directory:

```bash
export PROJECT_ROOT="$PWD"
salloc --partition=cpu --cpus-per-task=2 --mem=8G --time=01:00:00
cd "$PROJECT_ROOT"
bash baseline/setup_small_unity.sh cuda
# Prepare external inputs if they are not already available:
.baseline-envs/unet_small_cuda/bin/python -m voron_matsam.baseline download-data
.baseline-envs/unet_small_cuda/bin/python -m voron_matsam.baseline prepare
.baseline-envs/unet_small_cuda/bin/python -m baseline plan
exit
```

Skip downloading/preparation when the existing manifest is present; do not
overwrite an existing prepared directory. The setup is a minimal environment for
this model only, with PyTorch 2.6.0 CUDA 11.8 wheels and the common image-processing
dependencies. [PyTorch's official installation table](https://pytorch.org/get-started/previous-versions/)
lists this build. GPU driver compatibility must still be checked on the node.

Submit **one** job:

```bash
export PROJECT_ROOT="$PWD"
mkdir -p logs
sbatch --export=ALL baseline/small_unet.sbatch
```

This requests one GPU, two CPU cores, 8 GB host RAM, and 30 minutes. It does not
require an A100 or BF16 support. FP32 is intended for the available GTX 1080,
RTX 2080 and A100 hardware; actual device compatibility, VRAM use and runtime
are established by the first allocated run. Add your account flag as required.
An optional `--constraint=2080` selects that Unity GPU feature when available.

## CPU alternative

Install with `bash baseline/setup_small_unity.sh cpu` in a CPU allocation, then:

```bash
export PROJECT_ROOT="$PWD"
mkdir -p logs
sbatch --export=ALL baseline/small_unet_cpu.sbatch
```

Both templates evaluate validation images only and keep the same model and
training budget. For a direct run on a workstation or inside an allocation:

```bash
SMALL_DEVICE=cpu bash baseline/run_small.sh
```

Use `SMALL_PYTHON=/absolute/path/to/python` to reuse an existing environment.
`SMALL_OUTPUT` sets a new output directory; existing runs are never overwritten.

## What to inspect

Read `run.json`, `training_history.json`, `per_image.json`, and `summary.json` under
`outputs/small_unet/validation-JOBID/`. The checkpoint is `unet.pt`; integer instance
predictions are in `instances/`. Training metadata records elapsed training time,
parameter count, precision, device, optimizer updates and peak CUDA allocation
when running on a GPU. The requested host RAM is separate from GPU VRAM.

A decreasing semantic loss does not establish correct instance separation.
Inspect masks, PQ, false merges and false splits before expanding the budget.
Tune only on validation images. Keep the test holdout untouched until the model
and postprocessing choices are frozen. The suite summary excludes validation-only
runs, so this trial is not mixed with final benchmark results.

To increase training later, copy `baseline/configs/unet_small.json`, change its
epoch/step budget, and use `python -m baseline run --config YOUR_CONFIG` with a
new output directory. The architecture is based on [U-Net](https://arxiv.org/abs/1505.04597);
its reduced width and the watershed postprocessing are this experiment's settings.
