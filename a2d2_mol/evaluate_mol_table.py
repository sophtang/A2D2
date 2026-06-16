"""
Evaluate a finetuned molecule model checkpoint by sampling sequences
and computing metrics for the De Novo Small Molecule Generation table:
  Validity (%), Uniqueness (%), QED (↑), SA (↓), Quality (%), Diversity (↑), Sampling Time (↓)
"""

import os
import sys
import argparse
import time
import torch
import numpy as np
import pandas as pd
from tdc import Oracle, Evaluator

# add repo root (A2D2/) to sys.path so top-level packages like lightning_modules resolve
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from lightning_modules.any_length_remask import AnyOrderInsertionFlowModuleFT
from lightning_modules import AnyOrderInsertionFlowModule
from inference_quality_mol import sample_mol_eval
from mol_scoring.scoring_functions import MolScoringFunctions
from finetune_mol import MolFinetuner, get_tokenizer
from mol_utils.utils import str2bool, set_seed


def load_finetuned_model(checkpoint_path, pretrained_ckpt_path, device='cuda'):
    """Load a finetuned MolFinetuner from a Lightning checkpoint."""
    # We need to reconstruct the model the same way main() does, then load state
    # Load from Lightning checkpoint directly
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt.get('hyper_parameters', {})
    args = hparams.get('args', None)

    # Load pretrained base checkpoint to get config
    base_ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=False)
    if 'hyper_parameters' in base_ckpt:
        config = base_ckpt['hyper_parameters']['config']
    elif 'config' in base_ckpt:
        config = base_ckpt['config']
    else:
        raise ValueError("Cannot find config in base checkpoint")

    from omegaconf import OmegaConf, DictConfig
    if not OmegaConf.is_config(config):
        config = DictConfig(config)
    OmegaConf.set_struct(config, False)

    # Set adaptive schedule config from args or defaults
    config.training.use_adaptive_schedule = getattr(args, 'use_adaptive_schedule', True)
    config.training.schedule_hidden_dim = getattr(args, 'schedule_hidden_dim', 256)
    config.training.schedule_num_layers = getattr(args, 'schedule_num_layers', 2)
    config.training.schedule_loss_weight = getattr(args, 'schedule_loss_weight', 0.1)
    config.training.freeze_base_model = getattr(args, 'freeze_base_model', False)
    config.training.schedule_warmup_epochs = getattr(args, 'schedule_warmup_epochs', 0)
    config.training.use_bracket_safe = True
    OmegaConf.set_struct(config, True)

    # Determine if planner should be loaded based on disable_planner flag
    disable_planner = getattr(args, 'disable_planner', False)

    # Initialize policy model
    policy_model = AnyOrderInsertionFlowModuleFT(
        config=config,
        args=args,
        pretrained_checkpoint=pretrained_ckpt_path,
        insertion_planner=not disable_planner,
    )

    # Load policy model weights from the finetuned checkpoint
    state_dict = ckpt['state_dict']
    # Lightning wraps the model: 'policy_model.xxx' -> remove prefix for the sub-module
    policy_state = {}
    for k, v in state_dict.items():
        if k.startswith('policy_model.'):
            policy_state[k[len('policy_model.'):]] = v
    policy_model.load_state_dict(policy_state, strict=False)
    policy_model = policy_model.to(device)
    policy_model.eval()

    return policy_model, args, config


@torch.no_grad()
def evaluate_checkpoint(policy_model, tokenizer, reward_model, evaluator,
                        num_samples=1000, batch_size=50, max_length=256,
                        total_num_steps=256, quality_mode="both", num_remasking=2,
                        quality_threshold=0.5, unmask_quality_threshold=None, device='cuda'):
    """
    Sample `num_samples` molecules and compute all table metrics.
    Returns a dict with: validity, uniqueness, qed, sa, quality, diversity, sampling_time
    """
    all_valid_seqs = []
    all_smiles_generated = 0
    total_time = 0.0

    num_batches = (num_samples + batch_size - 1) // batch_size
    remaining = num_samples

    for b in range(num_batches):
        bs = min(batch_size, remaining)
        remaining -= bs

        t_start = time.time()
        result = sample_mol_eval(
            model=policy_model,
            reward_model=reward_model,
            tokenizer=tokenizer,
            steps=total_num_steps,
            mask=policy_model.interpolant.mask_token,
            pad=policy_model.interpolant.pad_token,
            batch_size=bs,
            max_length=max_length,
            quality_mode=quality_mode,
            num_remasking=num_remasking,
            quality_threshold=quality_threshold,
            unmask_quality_threshold=unmask_quality_threshold,
            evaluator=evaluator,
            dataframe=True,
        )
        t_end = time.time()

        # Unpack: uniqueSequences, qed, sa, valid_fraction, uniqueness, diversity, quality, df
        unique_seqs, qed_scores, sa_scores, valid_frac, uniq, div, qual, df = result
        
        all_valid_seqs.extend(list(unique_seqs) if not isinstance(unique_seqs, list) else unique_seqs)
        all_smiles_generated += bs
        total_time += (t_end - t_start)

        print(f"  Batch {b+1}/{num_batches}: {len(unique_seqs)} valid unique, "
              f"time={t_end - t_start:.1f}s")

    # --- Aggregate metrics over all samples ---
    total_generated = num_samples

    # Valid sequences (keeping duplicates for validity count)
    # Re-evaluate from scratch on all collected valid sequences
    all_unique = list(set(all_valid_seqs))
    num_valid = len(all_valid_seqs)  # total valid across batches (before dedup)
    num_unique = len(all_unique)

    validity = num_valid / total_generated * 100.0
    uniqueness = num_unique / num_valid * 100.0 if num_valid > 0 else 0.0

    # Diversity on unique SMILES
    diversity = evaluator(all_unique) if num_unique > 1 else 0.0

    # QED and SA on unique sequences
    if num_unique > 0:
        oracle_qed = Oracle('qed')
        oracle_sa = Oracle('sa')
        qed_vals = oracle_qed(all_unique)
        sa_vals = oracle_sa(all_unique)
        mean_qed = np.mean(qed_vals)
        mean_sa = np.mean(sa_vals)

        # Quality: unique sequences with QED >= 0.6 AND SA <= 4
        quality_mask = [(q >= 0.6 and s <= 4) for q, s in zip(qed_vals, sa_vals)]
        quality = sum(quality_mask) / total_generated * 100.0
    else:
        mean_qed = 0.0
        mean_sa = 0.0
        quality = 0.0

    sampling_time = total_time

    metrics = {
        'Validity (%)': validity,
        'Uniqueness (%)': uniqueness,
        'QED': mean_qed,
        'Synthetic Accessibility': mean_sa,
        'Quality (%)': quality,
        'Diversity': diversity,
        'Sampling Time (s)': sampling_time,
        'Num Generated': total_generated,
        'Num Valid': num_valid,
        'Num Unique': num_unique,
    }

    return metrics, all_unique, qed_vals if num_unique > 0 else [], sa_vals if num_unique > 0 else []


def main():
    parser = argparse.ArgumentParser(description="Evaluate a finetuned mol checkpoint")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the finetuned Lightning checkpoint (e.g., last.ckpt)')
    parser.add_argument('--pretrained_ckpt', type=str,
                        default=os.path.join(REPO_ROOT, 'pretrained', 'anylength_mol.ckpt'),
                        help='Path to the pretrained base model checkpoint '
                             '(defaults to <repo>/pretrained/anylength_mol.ckpt)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of molecules to sample')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for sampling')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--total_num_steps', type=int, default=256)
    parser.add_argument('--num_remasking', type=int, default=2)
    parser.add_argument('--disable_planner', action='store_true',
                        help='If set, disable remasking during evaluation (matches training mode)')
    parser.add_argument('--disable_insertion_planner', action='store_true',
                        help='If set, disable insertion quality filtering during evaluation')
    parser.add_argument('--disable_unmasking_planner', action='store_true',
                        help='If set, disable unmasking confidence planner during evaluation')
    parser.add_argument('--quality_threshold', type=float, default=0.5,
                        help='Threshold for insertion quality filtering during sampling')
    parser.add_argument('--unmask_quality_threshold', type=float, default=None,
                        help='If set, gate unmasking remasking on confidence: remask clean '
                             'tokens whose remasking_conf < threshold (overrides the '
                             'schedule-driven count). Default None = schedule-driven behavior.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results CSV. Defaults to checkpoint directory.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed, use_cuda=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Pretrained base: {args.pretrained_ckpt}")
    print(f"Disable planner (no remasking): {args.disable_planner}")
    print(f"Disable insertion planner: {args.disable_insertion_planner}")
    print(f"Disable unmasking planner: {args.disable_unmasking_planner}")

    policy_model, train_args, config = load_finetuned_model(
        args.checkpoint_path, args.pretrained_ckpt, device=device
    )

    tokenizer = get_tokenizer()
    score_func_names = ['qed', 'sa']
    reward_model = MolScoringFunctions(score_func_names, device=device)
    evaluator = Evaluator('diversity')

    use_remasking = not args.disable_planner
    disable_insertion_planner = args.disable_insertion_planner
    disable_unmasking_planner = args.disable_unmasking_planner

    # Map flags to quality_mode
    if args.disable_planner:
        quality_mode = "none"
    elif args.disable_insertion_planner and args.disable_unmasking_planner:
        quality_mode = "none"
    elif args.disable_insertion_planner:
        quality_mode = "unmasking_only"
    elif args.disable_unmasking_planner:
        quality_mode = "insertion_only"
    else:
        quality_mode = "both"

    print(f"\nSampling {args.num_samples} molecules (quality_mode={quality_mode})...")

    metrics, unique_smiles, qed_vals, sa_vals = evaluate_checkpoint(
        policy_model=policy_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        evaluator=evaluator,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        total_num_steps=args.total_num_steps,
        quality_mode=quality_mode,
        num_remasking=args.num_remasking,
        quality_threshold=getattr(args, 'quality_threshold', 0.5),
        unmask_quality_threshold=args.unmask_quality_threshold,
        device=device,
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("  De Novo Small Molecule Generation Results")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<30s}: {v:.4f}")
        else:
            print(f"  {k:<30s}: {v}")
    print("=" * 60)

    # Save results
    output_dir = args.output_dir or os.path.dirname(args.checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    if args.disable_planner:
        tag = "no_planner"
    elif args.disable_insertion_planner:
        tag = "no_insertion_planner"
    elif args.disable_unmasking_planner:
        tag = "no_unmasking_planner"
    else:
        tag = "with_planner"
    metrics_path = os.path.join(output_dir, f'eval_metrics_{tag}.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    if unique_smiles:
        smiles_path = os.path.join(output_dir, f'eval_smiles_{tag}.csv')
        df = pd.DataFrame({
            'SMILES': unique_smiles,
            'QED': qed_vals,
            'SA': sa_vals,
        })
        df.to_csv(smiles_path, index=False)
        print(f"SMILES saved to: {smiles_path}")


if __name__ == '__main__':
    main()
