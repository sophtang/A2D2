#!/bin/bash
#SBATCH --job-name=a2d2-pep-pretrain
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=512GB
#SBATCH --time=7-00:00:00
# SLURM's own catch-file (anything printed before the exec redirect below, plus
# slurm-infra messages). Relative to the submit dir, so submit this script from
# the a2d2_pep/ directory; the real run output is redirected via exec below.
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Pretrain the any-length insertion MDM on ~11M peptide SMILES on a dgx-b200 node.
# Submit with:  sbatch scripts/train_pep.sh   (from the a2d2_pep/ directory).
#
# DDP is launched by SLURM: one srun task per GPU. --gpus-per-node and
# --ntasks-per-node must match; change both together (and they override the
# training.devices value baked into config_pep.yaml via the hydra override below).

DATE=$(date +%Y%m%d)
SPECIAL_PREFIX='a2d2-peptide'

# Resolve a2d2_pep/ (which holds train.py + config_pep.yaml) so paths are
# repo-relative. This script lives in a2d2_pep/scripts/, so the direct-run
# fallback goes one level up. Under sbatch, BASH_SOURCE points at the spooled
# copy, so we rely on SLURM_SUBMIT_DIR (submit from the a2d2_pep/ directory).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$SCRIPT_DIR"

# Auto-detect GPUs from the SLURM allocation (falls back to 4 for `bash` runs).
DEVICES=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-4}}
NTASKS=${SLURM_NTASKS_PER_NODE:-$DEVICES}
NODES=${SLURM_NNODES:-1}

LOG_LOC="$SCRIPT_DIR/logs"
mkdir -p "$LOG_LOC/slurm"
exec > "${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_${SLURM_JOB_ID:-local}.log" 2>&1

# ---------------------------------------------------------------------------
# Weights & Biases: log in once on your machine before running this script with
#   `wandb login`  (or `export WANDB_API_KEY=<your-key>`).
# Do NOT hardcode your API key here. To disable W&B entirely, uncomment:
# export WANDB_MODE=disabled
# ---------------------------------------------------------------------------

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate the conda env that has the deps (torch / pytorch_lightning / hydra).
# The batch shell does NOT source ~/.bashrc, so conda is not on PATH. Override
# CONDA_ROOT to point at your conda/miniconda install, or just have `conda` on
# PATH; override CONDA_ENV if your env name differs from the one created by
# environment.yml.
CONDA_ENV="${CONDA_ENV:-a2d2}"
if [ -n "${CONDA_ROOT:-}" ]; then
    source "$CONDA_ROOT/bin/activate" "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/bin/activate" "$CONDA_ENV"
else
    echo "ERROR: conda not found; set CONDA_ROOT to your miniconda install." >&2
    exit 1
fi

# --- Distributed / NCCL setup (single node, intra-node NVLink) --------------
ETH_IFACE=$(ip -o -4 addr list | grep -v "127.0.0.1" | grep -E "ens|eth|enp|bond" | head -1 | awk '{print $2}')
if [ -z "$ETH_IFACE" ]; then
    ETH_IFACE=$(ip -o -4 addr list | grep -v "127.0.0.1" | grep -v "ibp" | head -1 | awk '{print $2}')
fi
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=$ETH_IFACE
export NCCL_P2P_LEVEL=NVL

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_NODELIST:-$(hostname)}" | head -n 1)
export MASTER_PORT=$(shuf -i 15000-59999 -n 1)
export NODE_RANK=${SLURM_NODEID:-0}

echo "=== a2d2 peptide pretraining (dgx-b200) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}  GPUs: $DEVICES  Tasks: $NTASKS"

# --task pep makes train.py load config_pep.yaml; the hydra overrides pin
# devices/nodes to the SLURM allocation so the two never drift apart.
srun --ntasks-per-node=$NTASKS python train.py --task pep \
    training.devices=$DEVICES \
    training.nodes=$NODES

conda deactivate
