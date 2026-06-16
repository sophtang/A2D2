#!/bin/bash
# NOTE: --partition and --qos below are specific to our cluster. Change them
# (or remove them and pass `--partition` on the `sbatch` command line) to match
# the partitions/QOS available on yours.
#SBATCH --job-name=peptide-finetune-len256
#SBATCH --partition=b200-mig90
#SBATCH --qos=mig
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80GB
#SBATCH --time=02-00:00:00
#SBATCH --output=logs/peptide_finetune_%A.log

# =====================================================================
# run_peptide_finetune.slurm
#
# Single-mode job (1 MIG GPU) running ONE finetune_quality (peptide)
# experiment. Select which mode to run via the MODE_ID variable below
# (or override at submit time with `sbatch --export=ALL,MODE_ID=2 ...`):
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

# Run prefix: YYYYMMDD + SLURM job ID
DATE_STAMP=$(date +%Y%m%d)
PREFIX="${DATE_STAMP}_job${SLURM_JOB_ID:-local$(date +%H%M%S)}"

# Default protein target (must be defined before path definitions below)
PROT_NAME=tfr

# --- Paths ------------------------------------------------------------
# Repo root is resolved at submit time so the script works from any clone:
#   - set A2D2_ROOT explicitly, OR
#   - run `sbatch` from the repo root (SLURM sets SLURM_SUBMIT_DIR), OR
#   - fall back to this script's location (a2d2_pep/scripts/ -> two levels up).
if [ -n "${A2D2_ROOT:-}" ]; then
    HOME_LOC="$A2D2_ROOT"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    HOME_LOC="$SLURM_SUBMIT_DIR"
else
    HOME_LOC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
SCRIPT_LOC="$HOME_LOC/a2d2_pep"
LOG_LOC="$HOME_LOC/logs"
SAVE_DIR="$HOME_LOC/checkpoints/finetune_test_peptides_${PROT_NAME}"
RESULTS_DIR="$HOME_LOC/results/peptide_test_ablation_${PROT_NAME}"

cd "$SCRIPT_LOC"

# BASE_PATH is passed as --base_path to finetune_quality.py: it's used
# to build the plot output path at $BASE_PATH/flexible/results/<run_name>
# (see finetune_quality.py:421). The pretrained checkpoint is now passed
# explicitly via --checkpoint_path below, so base_path no longer needs
# to follow the legacy /scratch layout.
BASE_PATH="${A2D2_BASE_PATH:-$HOME_LOC}"

mkdir -p "$LOG_LOC" "$SAVE_DIR" "$RESULTS_DIR"

# --- Environment setup ------------------------------------------------
# Do NOT hardcode your W&B key. Either `wandb login` once on the cluster,
# export WANDB_API_KEY in your shell/SLURM environment before submitting,
# or set WANDB_MODE=offline to skip logging entirely.
export WANDB_DIR=$HOME_LOC/.wandb
export WANDB_CONFIG_DIR=$HOME_LOC/.config/wandb
export WANDB_CACHE_DIR=$HOME_LOC/.cache/wandb
# Stop wandb from hijacking stdout/stderr (its default fd-redirect mode sends
# all output to wandb/run-*/files/output.log and freezes the RUN_LOG below).
# With console off, everything flows to the `>> "$RUN_LOG" 2>&1` redirect.
export WANDB_CONSOLE=off
mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

export TRITON_CACHE_DIR=$HOME_LOC/.triton/cache
mkdir -p "$TRITON_CACHE_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# Pretrained base checkpoint
PRETRAINED_CKPT="$HOME_LOC/pretrained/anylength_pep.ckpt"

# --- Shared training hyperparameters ----------------------------------
COMMON_ARGS=(
    --base_path "$BASE_PATH"
    --checkpoint_path "$PRETRAINED_CKPT"
    --prot_name "$PROT_NAME"
    --noise_removal
    --wdce_num_replicates 8
    --pool_size 100 
    --pool_refresh_fraction 1.0
    --buffer_size 50
    --batch_size 200
    --total_num_steps 256
    --num_iter 20
    --resample_every_n_step 10
    --num_epochs 1000
    --save_every_n_epochs 50
    --reset_every_n_step 1
    --alpha 0.1
    --no_mcts
    --schedule_warmup_epochs 20
    --alternation_frequency 5
    --num_remasking 3
    --quality_threshold 0.2
    --training_mini_batch_size 10
    --max_length 256
    --eval_every_n_epochs 50
    --min_peptide_bonds 4
    --grad_clip
    --seed 42
)

# --- Shared evaluation hyperparameters --------------------------------
EVAL_COMMON_ARGS=(
    --pretrained_ckpt "$PRETRAINED_CKPT"
    --num_samples 50
    --batch_size 200
    --max_length 256
    --total_num_steps 256
    --num_remasking 3
    --quality_threshold 0.2
    --prot_name "$PROT_NAME"
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

RUN_NAME="${PREFIX}_peptide_${PROT_NAME}_${MODE}"
RUN_LOG="$LOG_LOC/${RUN_NAME}.log"
RUN_SAVE_DIR="$SAVE_DIR/${RUN_NAME}"
RESULTS_SUBDIR="$RESULTS_DIR/${MODE}"
mkdir -p "$RUN_SAVE_DIR" "$RESULTS_SUBDIR"

echo "=== Peptide finetune (MODE_ID=$MODE_ID) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $SLURM_NODELIST"
echo "Mode: $MODE"
echo "Save dir: $RUN_SAVE_DIR"
echo "Results dir: $RESULTS_SUBDIR"
echo "Python: $PYTHON_EXECUTABLE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-(unset)}"

# =====================================================================
# Train
# =====================================================================
$PYTHON_EXECUTABLE $SCRIPT_LOC/finetune_quality.py \
    "${COMMON_ARGS[@]}" \
    --devices 1 \
    "${EXTRA_ARGS[@]}" \
    --save_path_dir "$RUN_SAVE_DIR" \
    >> "$RUN_LOG" 2>&1

echo "Training finished for $MODE. Log: $RUN_LOG"

# =====================================================================
# Evaluate
# =====================================================================
# finetune_quality.py saves to $RUN_SAVE_DIR/<auto_run_name>/last.ckpt,
# so glob the run_name subdir.
RUN_CKPT=$(ls -t "$RUN_SAVE_DIR"/*/last.ckpt 2>/dev/null | head -1)
if [ -z "$RUN_CKPT" ]; then
    echo "No checkpoint found in $RUN_SAVE_DIR — skipping eval."
    exit 1
fi

echo "Evaluating checkpoint: $RUN_CKPT"
$PYTHON_EXECUTABLE $SCRIPT_LOC/evaluate_peptide_table.py \
    --checkpoint_path "$RUN_CKPT" \
    "${EVAL_COMMON_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --output_dir "$RESULTS_SUBDIR" \
    --device cuda:0 \
    >> "$RESULTS_SUBDIR/${RUN_NAME}_eval.log" 2>&1

echo "Eval finished for $MODE. CSV: $RESULTS_SUBDIR/eval_metrics_${MODE}_${PROT_NAME}.csv"

conda deactivate
