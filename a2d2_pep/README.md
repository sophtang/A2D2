# A2D2 for Multi-Objective Therapeutic Peptide Generation 🧫

This part of the code fine-tunes an **any-length masked diffusion model (MDM)** over peptide SMILES with **A2D2** (Fine-Tuning Any-Length Discrete Diffusion for Adaptive Decoding) to optimize **five therapeutic properties simultaneously**: binding affinity to a target protein, solubility, non-hemolysis, non-fouling, and cell-membrane permeability.

A2D2 jointly fine-tunes the insertion and unmasking policies together with **insertion and unmasking quality predictors**, generating peptides via **Adaptive Joint Decoding (AJD)** that remasks low-quality tokens and drops low-quality insertions to sample from the reward-tilted distribution while preserving generation quality.

Peptides are represented as **SMILES** strings and tokenized with the SMILES Pair Encoding tokenizer (vocabulary size `V = 587`) from [PeptideCLM](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01443). Generated SMILES are decoded and validity-checked with the `SMILES2PEPTIDE` filter from [PepTune](https://arxiv.org/abs/2412.17780).

The codebase is partially built upon [FlexMDM (Kim et.al, 2025)](https://github.com/brianlck/FlexMDM/tree/main) and [TR2-D2 (Tang et.al, 2025)](https://github.com/sophtang/TR2-D2/tree/main).

## Environment Installation
```
# from the repository root
conda env create -f environment.yml

conda activate a2d2
```
The peptide scripts share the `a2d2` environment with the molecule and language experiments. See the root [`environment.yml`](../environment.yml) for the `flash-attn` install step.

## Model Pretrained Weights

A2D2 fine-tunes a pretrained any-length insertion MDM trained on ~11M peptide SMILES (7,451 sequences from CycPeptMPDB, 825,632 from SmProt, and ~10M modified peptides from CycloPs). Download the base checkpoint and place it at:
```
A2D2/pretrained/anylength_pep.ckpt
```
```bash
# from the repository root
pip install gdown
mkdir -p pretrained
gdown 1K8yxM-omh-MuPo0EG6UyxHZLk3HehoJc -O pretrained/anylength_pep.ckpt
```
(Or download manually from https://drive.google.com/file/d/1K8yxM-omh-MuPo0EG6UyxHZLk3HehoJc/view?usp=drive_link — a plain `wget`/`curl` of the link saves Google's HTML warning page, not the checkpoint.)
This is the default `--checkpoint_path`; pass `--checkpoint_path` to override it.

The reward classifiers (binding-affinity Transformer, plus XGBoost predictors for solubility, hemolysis, non-fouling, and permeability) and the SMILES PE tokenizer ship with the repo under [`pep_scoring/`](pep_scoring); no separate download is required. The PeptideCLM embedding model is fetched automatically from the Hugging Face Hub (`aaronfeller/PeptideCLM-23M-all`) on first run.

## Pretraining the Any-Length Model

If you only want to fine-tune with A2D2, download the released `anylength_pep.ckpt` above and skip this section. Follow these steps to reproduce the base checkpoint by pretraining the any-length insertion MDM from scratch.

### 1. Download the pretraining dataset

The pretraining corpus is ~11M peptide SMILES (7,451 from CycPeptMPDB, 825,632 from SmProt, and ~10M modified peptides from CycloPs), already tokenized with the in-repo SMILES PE tokenizer and saved as a Hugging Face `arrow` dataset (with `train`/`val` splits) via `save_to_disk`.

Download the archive and unpack it into [`data/`](data):

```bash
# from a2d2_pep/
pip install gdown
gdown https://drive.google.com/uc?id=1yCDr641WVjCtECg3nbG0nsMNu8j7d7gp -O 11M_peptide_smiles.zip
mkdir -p data
unzip 11M_peptide_smiles.zip -d data/
# result: a2d2_pep/data/11M_peptide_smiles/{train,val}/...
```

This is the default `training.data_path` in [`config_pep.yaml`](config_pep.yaml). To store the dataset elsewhere, set `training.data_path` (absolute, or relative to `a2d2_pep/`).

### 2. Configure

Pretraining is driven by [`config_pep.yaml`](config_pep.yaml). Key fields:

| Field | Default | Notes |
|-------|---------|-------|
| `training.data_path` | `data/11M_peptide_smiles` | Preprocessed arrow dataset from step 1. |
| `training.devices` | `4` | GPUs per node (DDP). |
| `training.batch_size` | `1024` | Global batch; gradient accumulation is derived automatically from `per_gpu_batch_size`. |
| `training.max_steps` | `1000000` | Total optimizer steps. |
| `training.learning_rate` | `3e-4` | AdamW LR with `warmup_steps: 2000`. |
| `training.checkpoint_dir` | `checkpoints/peptides` | A timestamped subdirectory is created per run. |
| `interpolant.max_length` | `1024` | Max token length. |

### 3. Pre-training Any-Length Peptide Model

Log in to Weights & Biases once (`wandb login`), or set `export WANDB_MODE=disabled` to skip logging. Then submit the SLURM job:

```bash
# from a2d2_pep/
sbatch train_pep.sh
```

`train_pep.sh` is a SLURM batch script that requests one `dgx-b200` node with 4 full B200 GPUs and launches DDP via `srun` (one task per GPU), running the equivalent of:

```bash
python train.py --task pep
```

It activates the conda env (`CONDA_ENV`, defaults to the `peptune` env) from `CONDA_ROOT` (defaults to the shared miniconda install) — the batch shell does not source `~/.bashrc`, so override these env vars if your install or env path differs. The GPU count is auto-detected from the SLURM allocation and passed to hydra as `training.devices`/`training.nodes`, so to scale just change `--gpus-per-node` and `--ntasks-per-node` together at the top of the script (they must match). `--task pep` makes `train.py` load `config_pep.yaml`.

Checkpoints are written to `checkpoints/peptides/<timestamp>/` (use `last.ckpt` / the best `train_loss` checkpoint as the `--checkpoint_path` for fine-tuning); the run log goes to `logs/<date>_a2d2-peptide_<jobid>.log` and SLURM's catch-file to `logs/slurm/`. To resume, add a `training.resume_path: /path/to/last.ckpt` entry to the config.

## Fine-Tune with A2D2

All paths resolve relative to the repository, so the scripts run from any checkout. Before running, create the output directories `A2D2/checkpoints`, `A2D2/results`, and `A2D2/logs` (the script also creates them on demand). Fine-tuning curves and a `<prot_name>_generation_results.csv` are written to `<base_path>/results/<run_name>/`, and checkpoints to `--save_path_dir`.

Choose a target protein with `--prot_name` (looked up in the built-in `PROTEINS` table — e.g. `glp1` for GLP-1R or `glast` for GLAST), or supply an arbitrary target with `--prot_seq <amino acid sequence>`.

#### Available `--prot_name` targets

The named targets and their amino-acid sequences are defined in the `PROTEINS` dict in [`finetune_quality.py`](finetune_quality.py) (search for `PROTEINS = {`). The default is `glast`; passing a name not in the table raises an error listing the valid keys. To add a new target, add a `'<key>': '<sequence>'` entry there, or skip the table entirely with `--prot_seq`.

| `--prot_name` | Target |
|---------------|--------|
| `tfr` | Transferrin receptor (TfR) |
| `glp1` | GLP-1 receptor (GLP-1R) |

### Single run

[`scripts/run_peptide_finetune.slurm`](scripts/run_peptide_finetune.slurm) runs a single `finetune_quality.py` experiment on one MIG GPU, then evaluates the resulting checkpoint. It bundles the hyperparameter set from the peptide column of the fine-tuning table in the paper — replicates `R = 8`, buffer size `B = 50`, resample interval `N_resample = 10`, gradient steps per iteration `N_update = 10`, alternation frequency `N_alt = 5`, warmup `N_warmup = 20`, sampling steps `N_steps = 256`, training mini-batch `10`, reward scaling `α = 0.1`, quality threshold `μ_min = 0.5`, and `--num_obj 5` — so you don't have to pass them by hand.

The script resolves the repo root automatically — `$A2D2_ROOT` if set, else the `sbatch` submit directory, else the script's own location — so either submit from the repo root or export your clone path. Set `CONDA_ROOT` (your miniconda install) and, if needed, `CONDA_ENV` (defaults to `peptune`) and `WANDB_ENTITY`:
```bash
export A2D2_ROOT=/path/to/your/A2D2     # absolute path to your clone
export CONDA_ROOT=/path/to/miniconda3   # or just have `conda` on PATH
export WANDB_ENTITY=your_wandb_entity   # optional
sbatch scripts/run_peptide_finetune.slurm
```

Select which variant to run with `MODE_ID` (default `0`): `0` = A2D2 (full planner), `1` = `--disable_planner`, `2` = `--disable_insertion_planner`, `3` = `--disable_unmasking_planner`. Override at submit time:
```bash
sbatch --export=ALL,MODE_ID=2 scripts/run_peptide_finetune.slurm
```
The target protein is set by the `PROT_NAME` variable near the top of the script (default `tfr`); edit it to one of the named targets above (or any key in the `PROTEINS` table). The pretrained base checkpoint is read from `$A2D2_ROOT/pretrained/anylength_pep.ckpt`. Outputs land in `checkpoints/finetune_test_peptides_<prot>/<job>_peptide_<prot>_<mode>/` and `results/peptide_test_ablation_<prot>/<mode>/`.

### Key arguments
- `--prot_name` / `--prot_seq` — target protein (named lookup, or a raw amino-acid sequence).
- `--alternation_frequency` — epochs to train each of {policy, planner} before alternating.
- `--alpha` — reward-tilting temperature (smaller = stronger reward optimization).
- `--buffer_size`, `--resample_every_n_step` — replay-buffer size and how often it is regenerated.

### Ablation flags
| Flag | Variant |
|------|---------|
| *(none)* | A2D2 w/ insertion + unmasking quality (alternation) |
| `--disable_planner` | A2D2 w/o quality (policy only, no remasking) |
| `--disable_insertion_planner` | A2D2 w/o insertion quality |
| `--disable_unmasking_planner` | A2D2 w/o unmasking/remasking quality |
| `--joint_training` | train policy + quality heads jointly (no alternation) |

During buffer generation only sequences passing the `SMILES2PEPTIDE` validity filter are retained; the scalarized multi-objective reward is added to the log Radon–Nikodym derivative of each sequence. Fine-tuning runs on a single GPU (`--devices 1`).

## Evaluation

Evaluation runs automatically every `--eval_every_n_epochs` epochs and at the end of training. It samples from the current model and reports the fraction of valid peptides along with the five therapeutic rewards (binding affinity, solubility, non-hemolysis, non-fouling, permeability), saving per-objective curves and `<prot_name>_generation_results.csv` under `<base_path>/results/<run_name>/`.

To resume a run, pass `--resume_ckpt /path/to/last.ckpt` (restores epoch, optimizer, and planner state; new checkpoints continue in the same directory).
