#!/bin/bash
# NOTE: --partition and --qos below are specific to our cluster. Change them
# (or remove them and pass `--partition` on the `sbatch` command line) to match
# the partitions/QOS available on yours.
#SBATCH --job-name=mol-finetune
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/slurm-%A.%x.log

# =====================================================================
# run_mol_finetune.slurm
#
# Single-mode job (1 MIG GPU) running ONE finetune_mol experiment.
# Select which mode to run via the MODE_ID variable below (or override
# at submit time with `sbatch --export=ALL,MODE_ID=2 ...`):
#   0) A2D2 (Ours)                – with full planner (alternating)
#   1) A2D2 w/o quality           – --disable_planner
#   2) A2D2 w/o insertion planner – --disable_insertion_planner
#   3) A2D2 w/o unmasking planner – --disable_unmasking_planner
#
# The job trains the selected mode then evaluates the resulting
# checkpoint on the same GPU.
# =====================================================================

set -e

# --- Mode selection ---------------------------------------------------
# Which experiment to run (0-3). Override with `--export=ALL,MODE_ID=N`.
MODE_ID="${MODE_ID:-0}"

# Run prefix
PREFIX=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}

# --- Paths ------------------------------------------------------------
# Repo root is resolved at submit time so the job runs from any clone:
#   - set A2D2_ROOT explicitly, OR
#   - submit with `sbatch` from the repo root (SLURM sets SLURM_SUBMIT_DIR;
#     note sbatch copies the script to a spool dir, so we can't rely on the
#     script's own path here), OR
#   - run the script directly, falling back to its location on disk.
if [ -n "${A2D2_ROOT:-}" ]; then
    HOME_LOC="$A2D2_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    HOME_LOC="$SLURM_SUBMIT_DIR"
else
    # This script lives in a2d2_mol/scripts/, so the repo root is two levels up.
    HOME_LOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
SCRIPT_LOC="$HOME_LOC/a2d2_mol"
LOG_LOC=$HOME_LOC/logs
SAVE_DIR=$HOME_LOC/checkpoints/finetune_mol
RESULTS_DIR=$HOME_LOC/results/mol_ablation

mkdir -p "$LOG_LOC" "$SAVE_DIR" "$RESULTS_DIR"

# --- Environment setup ------------------------------------------------
# Set WANDB_API_KEY in your shell/secret store before submitting (do NOT commit it):
#   export WANDB_API_KEY=...     or   `wandb login`
export WANDB_DIR=$HOME_LOC/.wandb
export WANDB_CONFIG_DIR=$HOME_LOC/.config/wandb
export WANDB_CACHE_DIR=$HOME_LOC/.cache/wandb
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

export TRITON_CACHE_DIR=$HOME_LOC/.triton/cache
mkdir -p "$TRITON_CACHE_DIR"

export TORCHINDUCTOR_CACHE_DIR=$HOME_LOC/.torchinductor/cache
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force unbuffered stdout/stderr so live training output is flushed to the
# redirected RUN_LOG (Python block-buffers stdout when it's a file, not a TTY).
export PYTHONUNBUFFERED=1

# Activate conda env. Override CONDA_ROOT to point at your conda/miniconda
# install, or just have `conda` on PATH; override CONDA_ENV if your env name
# differs from the one created by environment.yml.
CONDA_ENV="${CONDA_ENV:-a2d2}"
if [ -n "${CONDA_ROOT:-}" ]; then
    source "$CONDA_ROOT/bin/activate" "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/bin/activate" "$CONDA_ENV"
else
    echo "ERROR: conda not found; set CONDA_ROOT to your miniconda install." >&2
    exit 1
fi
PYTHON_EXECUTABLE=$(which python)

cd "$SCRIPT_LOC"

# Pretrained base checkpoint 
PRETRAINED_CKPT="$HOME_LOC/pretrained/anylength_mol.ckpt"

# --- Shared training hyperparameters ----------------------------------
COMMON_ARGS=(
    --base_path "$HOME_LOC"
    --use_quality_filter
    --noise_removal
    --wdce_num_replicates 16
    --pool_size 1000
    --pool_refresh_fraction 0.3
    --buffer_size 100
    --batch_size 200
    --training_mini_batch_size 20
    --max_length 256
    --total_num_steps 256
    --num_iter 20
    --resample_every_n_step 10
    --num_epochs 1000
    --save_every_n_epochs 100
    --reset_every_n_step 1
    --alpha 0.01
    --no_mcts
    --schedule_warmup_epochs 20
    --alternation_frequency 5
    --num_remasking 3
    --quality_threshold 0.3
    --checkpoint_path "$PRETRAINED_CKPT"
    --grad_clip
    --qed_only
    --seed 42
    --num_training_steps_per_epoch 25
)

# --- Shared evaluation hyperparameters --------------------------------
EVAL_COMMON_ARGS=(
    --pretrained_ckpt "$PRETRAINED_CKPT"
    --num_samples 1000
    --batch_size 50
    --max_length 256
    --total_num_steps 256
    --num_remasking 2
    --quality_threshold 0.3
    --seed 42
)

# =====================================================================
# Pick experiment from $MODE_ID
# =====================================================================
case "$MODE_ID" in
    0) MODE="with_planner";         EXTRA_ARGS=() ;;
    1) MODE="no_planner";           EXTRA_ARGS=(--disable_planner) ;;
    2) MODE="no_insertion_planner"; EXTRA_ARGS=(--disable_insertion_planner) ;;
    3) MODE="no_unmasking_planner"; EXTRA_ARGS=(--disable_unmasking_planner) ;;
    *) echo "Unknown MODE_ID=$MODE_ID (expected 0-3)"; exit 1 ;;
esac

RUN_NAME="${PREFIX}_mol_${MODE}"
RUN_LOG="$LOG_LOC/${RUN_NAME}.log"
RUN_SAVE_DIR="$SAVE_DIR/${RUN_NAME}"
RESULTS_SUBDIR="$RESULTS_DIR/${MODE}"
mkdir -p "$RUN_SAVE_DIR" "$RESULTS_SUBDIR"

echo "=== Mol finetune (MODE_ID=$MODE_ID) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $SLURM_NODELIST"
echo "Mode: $MODE"
echo "Save dir: $RUN_SAVE_DIR"
echo "Results dir: $RESULTS_SUBDIR"
echo "Python: $PYTHON_EXECUTABLE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(unset)}"

# =====================================================================
# Train
# =====================================================================
$PYTHON_EXECUTABLE $SCRIPT_LOC/finetune_mol.py \
    "${COMMON_ARGS[@]}" \
    --devices 1 \
    "${EXTRA_ARGS[@]}" \
    --save_path_dir "$RUN_SAVE_DIR" \
    >> "$RUN_LOG" 2>&1

echo "Training finished for $MODE. Log: $RUN_LOG"

# =====================================================================
# Evaluate
# =====================================================================
RUN_CKPT=$(ls -t "$RUN_SAVE_DIR"/*/last.ckpt "$RUN_SAVE_DIR"/last.ckpt 2>/dev/null | head -1)
if [ -z "$RUN_CKPT" ]; then
    echo "No checkpoint found in $RUN_SAVE_DIR — skipping eval."
    exit 1
fi

echo "Evaluating checkpoint: $RUN_CKPT"
$PYTHON_EXECUTABLE $SCRIPT_LOC/evaluate_mol_table.py \
    --checkpoint_path "$RUN_CKPT" \
    "${EVAL_COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --output_dir "$RESULTS_SUBDIR" \
    --device cuda:0 \
    >> "$RESULTS_SUBDIR/eval.log" 2>&1

echo "Eval finished for $MODE. CSV: $RESULTS_SUBDIR/eval_metrics_${MODE}.csv"

conda deactivate
