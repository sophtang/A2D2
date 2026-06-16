# A2D2 for Molecule Generation 🧪

This part of the code fine-tunes an **any-length masked diffusion model (MDM)** over molecules with **A2D2** (Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding) to optimize drug-likeness rewards (QED, and optionally synthetic accessibility / SA).

A2D2 jointly fine-tunes the insertion and unmasking policies together with **insertion and unmasking quality predictors**, generating molecules via **Adaptive Joint Decoding (AJD)** that remasks low-quality tokens and drops low-quality insertions to sample from the reward-tilted distribution while preserving generation quality.

Molecules are represented as [SAFE](https://github.com/datamol-io/safe) strings and tokenized with the `datamol-io/safe-gpt` tokenizer.

The codebase is partially built upon [FlexMDM (Kim et.al, 2025)](https://github.com/brianlck/FlexMDM/tree/main) and [TR2-D2 (Tang et.al, 2025)](https://github.com/sophtang/TR2-D2/tree/main).

## Environment Installation
```
# from the repository root
conda env create -f environment.yml

conda activate a2d2
```
The molecule scripts share the `a2d2` environment with the peptide and language experiments. See the root [`environment.yml`](../environment.yml) for the `flash-attn` install step.

## Model Pretrained Weights

A2D2 fine-tunes a pretrained any-length insertion MDM trained on drug-like SAFE molecules. Download the base checkpoint and place it at:
```
A2D2/pretrained/anylength_mol.ckpt
```
```bash
# from the repository root
pip install gdown
mkdir -p pretrained
gdown 1I5EGiV1I5XZZpB9JAKABFLKVqfCyenxq -O pretrained/anylength_mol.ckpt
```
(Or download manually from https://drive.google.com/file/d/1I5EGiV1I5XZZpB9JAKABFLKVqfCyenxq/view?usp=drive_link — a plain `wget`/`curl` of the link saves Google's HTML warning page, not the checkpoint.)
This is the default `--checkpoint_path` (for fine-tuning) and `--pretrained_ckpt` (for evaluation) used throughout.

## Pretraining the Any-Length Model

If you only want to fine-tune with A2D2, download the released `anylength_mol.ckpt` above and skip this section. Follow these steps to reproduce the base checkpoint by pretraining the any-length insertion MDM from scratch.

### 1. The pretraining dataset

The model is pretrained on drug-like [SAFE](https://github.com/datamol-io/safe) molecules from the [`datamol-io/safe-gpt`](https://huggingface.co/datasets/datamol-io/safe-gpt) dataset (~1.1B molecules) on the Hugging Face Hub. **No manual download is required** — the dataset is loaded in streaming mode (`load_dataset(..., streaming=True)`) and tokenized on the fly with the `datamol-io/safe-gpt` tokenizer, both fetched automatically on first run.

The dataset is configured in [`config_mol.yaml`](config_mol.yaml):

```yaml
hf_dataset:
  name: "datamol-io/safe-gpt"
  smiles_column: "smiles"
```

To pretrain on a different Hugging Face SMILES/SAFE dataset, change `hf_dataset.name` (and `smiles_column` to match its column).

### 2. Configure

Pretraining is driven by [`config_mol.yaml`](config_mol.yaml). Key fields:

| Field | Default | Notes |
|-------|---------|-------|
| `hf_dataset.name` | `datamol-io/safe-gpt` | Streaming HF dataset (auto-downloaded). |
| `training.devices` | `2` | GPUs per node (DDP). |
| `training.batch_size` | `2048` | Global batch; gradient accumulation is derived automatically from `per_gpu_batch_size`. |
| `training.max_steps` | `500000` | Total optimizer steps. |
| `training.learning_rate` | `3e-4` | AdamW LR with `warmup_steps: 2000`. |
| `training.save_every_n_steps` | `1000` | Step-based checkpointing (used for streaming datasets). |
| `training.checkpoint_dir` | `checkpoints/pretrain_mol` | A timestamped subdirectory is created per run. |
| `interpolant.max_length` | `256` | Max token length. |

### 3. Pre-training Any-Length Molecule Model

Log in to Weights & Biases once (`wandb login`), or set `export WANDB_MODE=disabled` to skip logging. Then submit the SLURM job:

```bash
# from a2d2_mol/
sbatch train_mol.sh
```

`train_mol.sh` is a SLURM batch script that requests one `dgx-b200` node with 2 full B200 GPUs and launches DDP via `srun` (one task per GPU), running the equivalent of:

```bash
python train.py --task mol
```

It activates the conda env (`CONDA_ENV`, defaults to the `peptune` env) from `CONDA_ROOT` (defaults to the shared miniconda install) — the batch shell does not source `~/.bashrc`, so override these env vars if your install or env path differs. The GPU count is auto-detected from the SLURM allocation and passed to hydra as `training.devices`/`training.nodes`, so to scale just change `--gpus-per-node` and `--ntasks-per-node` together at the top of the script (they must match). `--task mol` makes `train.py` load `config_mol.yaml`.

Checkpoints are written to `checkpoints/pretrain_mol/<timestamp>/` (use `last.ckpt` / the best `train_loss` checkpoint as the `--checkpoint_path` / `--pretrained_ckpt` for fine-tuning and evaluation); the run log goes to `logs/<date>_a2d2-mol_<jobid>.log` and SLURM's catch-file to `logs/slurm/`. To resume, add a `training.resume_path: /path/to/last.ckpt` entry to the config.

## Fine-Tune with A2D2

The canonical run directory is the parent `a2d2/` package (`finetune_mol.py`, `inference_quality_mol.py`, `sampling.py`, and `mol_scoring/` here are the molecule-specific modules used from there). Before running:

1. Set `--base_path` to the location of `a2d2`. Results plots are written to `<base_path>/flexible/results/<run_name>/`.
2. Create the output directories: `a2d2/checkpoints/finetune_mol`, `a2d2/results`, and `a2d2/logs`.

### Single run

[`scripts/run_mol_finetune.slurm`](scripts/run_mol_finetune.slurm) runs a single `finetune_mol.py` experiment on one MIG GPU, then evaluates the resulting checkpoint. It bundles the full hyperparameter set used in the paper (replicates `R = 16`, pool size `1000`, buffer size `100`, sampling steps `N_steps = 90`, warmup `N_warmup = 20`, alternation frequency `N_alt = 5`, reward scaling `α = 0.01`, quality threshold `μ_min = 0.3`, `--qed_only`), so you don't have to pass them by hand.

The script resolves the repo root automatically — `$A2D2_ROOT` if set, else the `sbatch` submit directory, else the script's own location — so either submit from the repo root or export your clone path. Set `CONDA_ROOT` (your miniconda install) and, if needed, `CONDA_ENV` (defaults to `peptune`):
```bash
export A2D2_ROOT=/path/to/your/A2D2     # absolute path to your clone
export CONDA_ROOT=/path/to/miniconda3   # or just have `conda` on PATH
sbatch scripts/run_mol_finetune.slurm
```

Select which variant to run with `MODE_ID` (default `0`): `0` = A2D2 (full planner), `1` = `--disable_planner`, `2` = `--disable_insertion_planner`, `3` = `--disable_unmasking_planner`. Override at submit time:
```bash
sbatch --export=ALL,MODE_ID=2 scripts/run_mol_finetune.slurm
```
The pretrained base checkpoint is read from `$A2D2_ROOT/pretrained/anylength_mol.ckpt`. Outputs land in `checkpoints/finetune_mol/<job>_mol_<mode>/` and `results/mol_ablation/<mode>/`.

### Ablation flags
| Flag | Variant |
|------|---------|
| *(none)* | A2D2 w/ insertion + unmasking quality (alternation) |
| `--disable_planner` | A2D2 w/o quality (policy only, no remasking) |
| `--disable_insertion_planner` | A2D2 w/o insertion quality |
| `--disable_unmasking_planner` | A2D2 w/o unmasking/remasking quality |
| `--joint_training` | train policy + quality heads jointly (no alternation) |

## Evaluation

Evaluation runs automatically at the end of the SLURM job. To evaluate a checkpoint manually:
```
python evaluate_mol_table.py \
    --checkpoint_path /path/to/a2d2/checkpoints/finetune_mol/my_run/last.ckpt \
    --pretrained_ckpt /path/to/A2D2/pretrained/anylength_mol.ckpt \
    --output_dir /path/to/results \
    --num_samples 1000 --batch_size 50 \
    --max_length 256 --total_num_steps 256 \
    --num_remasking 2 --quality_threshold 0.3 --seed 42 --device cuda:0
```
This reports QED, SA, validity, uniqueness, diversity, and mean unmasking/insertion quality over the generated molecules and writes `eval_metrics_<mode>.csv`.
