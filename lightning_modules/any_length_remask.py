import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch.nn.functional as F
from model.transformer import AnyOrderMaskInsertionFlow
from model.interpolant import AnyOrderMaskInsertionInterpolant, ModelPrediction
from .bregman import jump_kernel_elbo, mse
from .schedule import get_schedule_from_config
from lightning_modules.any_order import AnyOrderInsertionFlowModule
from model.model_wrapper import RemaskingAnyOrder
from sampling import _sample_tokens

import re
from typing import Dict, Any
from dataclasses import dataclass

def strip_orig_mod_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a new state_dict where any key containing '._orig_mod.' is replaced
    by removing the '_orig_mod' segment, e.g.
      'model._orig_mod.vocab_embed.embedding'
    becomes
      'model.vocab_embed.embedding'
    """
    new_state_dict: Dict[str, Any] = {}
    for key, value in state_dict.items():
        # remove all occurrences of '._orig_mod.'
        clean_key = re.sub(r"\._orig_mod\.", ".", key)
        new_state_dict[clean_key] = value
    return new_state_dict


@torch.no_grad()
def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUROC (Mann-Whitney U statistic).

    AUC = P(score[pos] > score[neg]); 0.5 means no discrimination. Returns NaN
    when only one class is present (AUC undefined). Ties are not averaged, which
    is fine for continuous logits used here.
    """
    scores = scores.float().reshape(-1)
    labels = labels.float().reshape(-1)
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=scores.dtype)
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc.item()


class AnyOrderInsertionFlowModuleFT(AnyOrderInsertionFlowModule):
    """
    Wrapper around AnyOrderInsertionFlowModule that adds adaptive schedule model
    for fine-tuning. Can load a pretrained AnyOrderInsertionFlowModule checkpoint
    and add the schedule model on top.
    """
    def __init__(self, config, args, pretrained_checkpoint, insertion_planner=False):
        # Initialize parent class first
        super().__init__(config)
        
        self.args = args
        self.insertion_planner = insertion_planner
        
        # Save hyperparameters for this class (overrides parent's save)
        self.save_hyperparameters(ignore=['pretrained_checkpoint', 'args'])
        
        # Load pretrained model weights BEFORE initializing planner to avoid circular reference
        if pretrained_checkpoint is not None:
            self.load_pretrained_model(pretrained_checkpoint)
        
        # Initialize adaptive schedule model AFTER loading pretrained weights
        self.planner = RemaskingAnyOrder(
            backbone=self,
            d_model=self.config.model.hidden_size,
            insertion_planner=insertion_planner)
        
    def load_pretrained_model(self, checkpoint_path: str):
        """
        Load pretrained AnyOrderInsertionFlowModule weights.
        Only loads the base model and interpolant, not the schedule model.
        """
        print(f"Loading pretrained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract state dict - handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Strip _orig_mod keys if present
        state_dict = strip_orig_mod_keys(state_dict)
        
        # Filter out planner keys (if any exist from a previous FT checkpoint)
        base_state_dict = {k: v for k, v in state_dict.items() 
                          if not k.startswith('planner.')}
        
        # Load the base model weights
        # Use strict=False to ignore missing schedule_model keys
        incompatible_keys = self.load_state_dict(base_state_dict, strict=False)
        
        # Filter out expected missing planner keys for cleaner output
        unexpected_missing = [k for k in incompatible_keys.missing_keys 
                            if not k.startswith('planner.')]
        planner_missing = [k for k in incompatible_keys.missing_keys 
                          if k.startswith('planner.')]
        
        if unexpected_missing:
            print(f"Warning: Unexpected missing keys from pretrained checkpoint: {unexpected_missing}")
        if planner_missing:
            print(f"Note: Planner will be trained from scratch ({len(planner_missing)} parameters)")
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys in pretrained checkpoint: {incompatible_keys.unexpected_keys}")
        
        # Freeze base model if specified
        if self.config.training.get('freeze_base_model', False):
            print("Freezing base model parameters")
            for name, param in self.named_parameters():
                if not name.startswith('planner.'):
                    param.requires_grad = False

    def forward(self, x, t, return_features=False):
        # Use parent class forward method
        return super().forward(x, t, return_features=return_features)

    def training_loss(self, x1, t):
        # Use parent class training_loss for base model loss
        # Planner is trained separately via loss_planner_flexible with reward gradients
        unmask_loss, insertion_loss, total_loss = super().training_loss(x1, t)
        return unmask_loss, insertion_loss, total_loss
    
    
    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        t = self.sample_time(x1.shape[0], x1.device)

        # Calculate the base model loss (planner trained separately, not here)
        unmask_loss, len_loss, loss = self.training_loss(x1, t)
        
        # Log component losses
        self.log("train/unmask_loss", unmask_loss, prog_bar=True)
        self.log("train/len_loss", len_loss, prog_bar=True)
        self.log("train/total_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        t = self.sample_time(x1.shape[0], x1.device)
        unmask_loss, len_loss, loss = self.training_loss(x1, t)

        self.log("val/unmask_loss", unmask_loss, prog_bar=True, sync_dist=True)
        self.log("val/len_loss", len_loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        return loss
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, strict=True, **kwargs):
        """
        Custom checkpoint loading that handles finetuned checkpoints wrapped by PeptideFinetuner.
        Extracts config from original pretrained checkpoint and loads finetuned weights.
        """
        print(f"Loading finetuned checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location or 'cpu', weights_only=False)
        
        # Check if this is a wrapped checkpoint (from PeptideFinetuner)
        hparams = checkpoint.get('hyper_parameters', {})
        state_dict = checkpoint.get('state_dict', {})
        
        # Check for policy_model prefix in state_dict (indicates PeptideFinetuner wrapper)
        has_policy_prefix = any(k.startswith('policy_model.') for k in state_dict.keys())
        
        if has_policy_prefix:
            # Detect model type (molecule vs peptide) based on vocab size in checkpoint
            # Molecule models have vocab size ~1882, peptide models have ~587
            vocab_size = None
            for k, v in state_dict.items():
                if 'vocab_embed.embedding' in k:
                    vocab_size = v.shape[0]
                    break
            
            is_molecule_model = vocab_size is not None and vocab_size > 1000
            model_type = "MolFinetuner" if is_molecule_model else "PeptideFinetuner"
            print(f"Detected wrapped finetuned checkpoint ({model_type}, vocab_size={vocab_size})")
            
            # Extract args from hyperparameters
            if 'args' not in hparams:
                raise ValueError(f"Cannot find 'args' in hyperparameters. This checkpoint may not be from {model_type}.")
            
            args = hparams['args']
            print(f"Found args in hyperparameters, type: {type(args)}")
            
            # Get original checkpoint path from args
            # Handle both Namespace (hasattr) and dict (get) access patterns
            original_ckpt_path = None
            if hasattr(args, 'checkpoint_path'):
                original_ckpt_path = args.checkpoint_path
            elif isinstance(args, dict) and 'checkpoint_path' in args:
                original_ckpt_path = args['checkpoint_path']
            
            # If checkpoint_path is not set or is None, use default pretrained checkpoint
            # Select appropriate default based on detected model type
            if original_ckpt_path is None:
                _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if is_molecule_model:
                    original_ckpt_path = os.path.join(_repo_root, 'pretrained', 'anylength_mol.ckpt')
                    print(f"Warning: checkpoint_path not found in args, using default molecule pretrained checkpoint")
                else:
                    original_ckpt_path = os.path.join(_repo_root, 'pretrained', 'anylength_pep.ckpt')
                    print(f"Warning: checkpoint_path not found in args, using default peptide pretrained checkpoint")
            
            # Try to load config directly from checkpoint first (new checkpoints)
            # Fall back to loading from original checkpoint (old checkpoints)
            if 'config' in checkpoint:
                print("Found config directly in checkpoint")
                config = checkpoint['config']
            else:
                print(f"Config not in checkpoint, loading from original checkpoint: {original_ckpt_path}")
                
                # Load config from original pretrained checkpoint
                orig_ckpt = torch.load(original_ckpt_path, map_location='cpu', weights_only=False)
                if 'config' not in orig_ckpt:
                    raise ValueError(f"Original checkpoint {original_ckpt_path} does not contain config")
                
                config = orig_ckpt['config']
            
            # Ensure adaptive schedule is enabled
            # Need to disable struct mode to add new keys to OmegaConf config
            from omegaconf import OmegaConf
            if hasattr(config, 'training'):
                OmegaConf.set_struct(config, False)
                config.training.use_adaptive_schedule = True
                OmegaConf.set_struct(config, True)
            
            # Create args object if needed
            if not hasattr(args, '__dict__'):
                # Convert dict to object with attributes
                class Args:
                    pass
                args_obj = Args()
                for k, v in args.items():
                    setattr(args_obj, k, v)
                args = args_obj
            
            # Initialize model with config and args
            model = cls(
                config=config,
                args=args,
                pretrained_checkpoint=None,  # Don't reload pretrained, weights already in checkpoint
                insertion_planner=getattr(args, 'insertion_planner', False)
            )
            
            # Extract policy_model weights from state_dict
            policy_state = {}
            for k, v in state_dict.items():
                if k.startswith('policy_model.'):
                    # Strip 'policy_model.' prefix
                    new_key = k[len('policy_model.'):]
                    policy_state[new_key] = v
            
            # Load the finetuned weights
            incompatible = model.load_state_dict(policy_state, strict=False)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                print(f"Warning: Incompatible keys when loading finetuned weights:")
                if incompatible.missing_keys:
                    print(f"  Missing: {incompatible.missing_keys[:5]}...")
                if incompatible.unexpected_keys:
                    print(f"  Unexpected: {incompatible.unexpected_keys[:5]}...")
            
            # Initialize or load EMA params
            if model.use_ema:
                if "ema_params" in checkpoint:
                    # Load EMA params from checkpoint
                    model.ema_params = checkpoint["ema_params"]
                    print("Loaded EMA params from checkpoint")
                else:
                    # Initialize empty EMA params (will be populated if needed)
                    model.ema_params = {
                        name: param.clone().detach()
                        for name, param in model.named_parameters()
                    }
                    print("Initialized EMA params from current model state")
            else:
                model.ema_params = {}
            
            # Load planner state if it exists
            if "planner_state" in checkpoint and hasattr(model, 'planner'):
                model.planner.load_state_dict(checkpoint["planner_state"], strict=False)
                print("Loaded planner state from checkpoint")
            
            return model
        else:
            # Not a wrapped checkpoint, use default Lightning loading
            # But we still need to provide required __init__ arguments
            raise NotImplementedError(
                "Direct finetuned checkpoints (not wrapped by PeptideFinetuner) are not yet supported. "
                "Please provide config and args as kwargs."
            )
    
    def on_save_checkpoint(self, checkpoint):
        """Save config and EMA params, including planner state."""
        # Call parent to save config and base model EMA
        super().on_save_checkpoint(checkpoint)
        
        # Explicitly save planner state
        if hasattr(self, 'planner'):
            checkpoint["planner_state"] = self.planner.state_dict()
    
    def on_load_checkpoint(self, checkpoint):
        """Load config and reinitialize interpolant, including planner."""
        # For finetuned checkpoints loaded via custom load_from_checkpoint,
        # config may not be in checkpoint (it's loaded from original checkpoint)
        if "config" in checkpoint:
            # Call parent to restore config and interpolant
            super().on_load_checkpoint(checkpoint)
        else:
            # Config already set during __init__ via load_from_checkpoint
            # Just restore EMA params if they exist
            if self.use_ema and "ema_params" in checkpoint:
                self.ema_params = checkpoint["ema_params"]
        
        # Restore planner state if it exists in checkpoint
        if hasattr(self, 'planner') and "planner_state" in checkpoint:
            self.planner.load_state_dict(checkpoint["planner_state"])
            print("Loaded planner from checkpoint")
            
    def loss_wdce_flexible(self, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, eps=1e-3, centering=False, centering_strength=1.0, softmax_temperature=1.0):
        r"""
        Weighted denoising cross entropy loss
        X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
        
        log_rnd: [B] — pre-computed importance weights (already softmax-normalized over the full buffer)
        x: [B, L] (no mask)
        num_replicates: R, number of replicates of each row in x
        weight_func: w(lambda) for each sample, 1/lambda by default
        centering_strength: float, controls how much of the mean is subtracted (DMPO-style)
        softmax_temperature: float, temperature for softmax on log_rnd (>1 smooths weights)
        """
        
        batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, L]
        
        batch_weights = (log_rnd.detach() / softmax_temperature).softmax(dim=-1)  # [B]
        if centering:
            batch_weights = batch_weights - centering_strength * batch_weights.mean()
        
        batch_weights = batch_weights.repeat_interleave(num_replicates, dim=0)
        
        lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
        lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
        
        t = lamda
        
        # compute unmasking and insertion loss
        interpolant_sample = self.interpolant.sample_interpolant(t, batch)
        unmask_weight, insert_weight = self.interpolant.elbo_weight(t, batch)

        prediction: ModelPrediction = self(interpolant_sample.xt, t)

        scale_factor = self.config.interpolant.max_length

        match self.unmask_loss_fn:
            case "elbo":
                mask_indices = interpolant_sample.mask_indices
                unmask_loss_all = torch.zeros_like(unmask_weight)  # [B*R, L]
                unmask_loss_all[mask_indices] = unmask_weight[mask_indices] * F.cross_entropy(
                    prediction.token_logits[mask_indices],
                    interpolant_sample.unmasked[mask_indices],
                    reduction="none",
                )
                unmask_loss = unmask_loss_all.sum(dim=1) / scale_factor  # [B*R]
            case _:
                raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_fn}")

        match self.insert_loss_fn:
            case "expectation":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss_all = torch.zeros_like(insert_weight)  # [B*R, L+1]
                insertion_loss_all[gaps_mask] = insert_weight[gaps_mask] * jump_kernel_elbo(
                    gaps[gaps_mask], prediction.expected_gaps[gaps_mask]
                )
                insertion_loss = insertion_loss_all.sum(dim=1) / scale_factor  # [B*R]

            case "distribution":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss_all = torch.zeros_like(insert_weight)  # [B*R, L+1]
                insertion_loss_all[gaps_mask] = insert_weight[gaps_mask] * F.cross_entropy(
                    prediction.length_posterior[gaps_mask], gaps[gaps_mask]
                )
                insertion_loss = insertion_loss_all.sum(dim=1) / scale_factor  # [B*R]

        total_loss = unmask_loss + insertion_loss  # [B*R]
        # end compute unmasking and insertion loss
        
        weighted_loss = total_loss * batch_weights  # [B*R]
        return weighted_loss.mean()
    
    def one_step_sampler(self, xt, t, pred_rate=None):
        """
        Sample one step of unmasking using model predictions.
        
        Args:
            xt: Current state [B, L]
            t: Time [B]
            pred_rate: Optional pre-computed ModelPrediction. If None, will compute from model.
        
        Returns:
            new_xt: Next state [B, L]
            update_ids: Boolean mask of updated positions [B, L]
        """
        mask = self.interpolant.mask_token
        pad = self.interpolant.pad_token
        batch_size, L = xt.shape
        device = xt.device
        steps = self.args.total_num_steps
        dt = 1.0 / steps
        max_length = self.interpolant.max_length
        # Use actual tensor dimension L instead of max_length to handle replicated batches
        batch_idx_L = (
            torch.arange(batch_size, device=device)
            .view(batch_size, 1)
            .expand(batch_size, L)
        )
        pos_idx_L = (
            torch.arange(L, device=device)
            .view(1, L)
            .expand(batch_size, L)
        )
        
        # ——— predict and convert rates ———
        if pred_rate is None:
            pred_rate = self(xt, t)
        pred_rate = self.interpolant.to_actual_rate(xt, pred_rate, t)
        unmask_rate = pred_rate.unmask_rate  # (B, L, V)
        len_rate = pred_rate.length_rate  # (B, L+1)

        # ——— unmask step (Euler) ———
        mask_pos = (xt == self.interpolant.mask_token).nonzero(as_tuple=True)
        unmask_rate[xt != mask] = 0
        unmask_rate[mask_pos + (mask,)] = 0
        unmask_rate[mask_pos + (mask,)] = -unmask_rate[mask_pos + (slice(None),)].sum(dim=1)
        trans_prob = (unmask_rate * dt).clamp(0.0, 1.0)
        
        # add "stay" probability
        _xt = xt.clone()
        _xt[xt == pad] = mask
        trans_prob.scatter_add_(
            2,
            _xt.unsqueeze(-1),
            torch.ones_like(_xt.unsqueeze(-1), dtype=trans_prob.dtype),
        )

        trans_prob[mask_pos + (mask,)] = 0.0  # remove mask token from sampling at the last step
        
        # Renormalize probabilities to ensure they sum to 1
        prob_sum = trans_prob[mask_pos].sum(dim=-1, keepdim=True)
        # Avoid division by zero; if all probs are 0, use uniform distribution (excluding mask and pad)
        mask_has_zero_prob = (prob_sum.squeeze(-1) == 0.0)
        if mask_has_zero_prob.any():
            # Create uniform distribution over valid tokens (excluding mask and pad)
            num_zero_prob = mask_has_zero_prob.sum().item()
            uniform_prob = torch.zeros((num_zero_prob, trans_prob.shape[-1]), device=device, dtype=trans_prob.dtype)
            uniform_prob[:, :mask] = 1.0 / mask  # Uniform over tokens 0 to mask-1
            trans_prob[mask_pos[0][mask_has_zero_prob], mask_pos[1][mask_has_zero_prob]] = uniform_prob
        else:
            # Normalize to sum to 1
            trans_prob[mask_pos] = trans_prob[mask_pos] / prob_sum

        new_xt = _sample_tokens(trans_prob)
        new_xt[xt == pad] = pad
        new_xt = torch.where((xt != mask) & (xt != pad), xt, new_xt)
       
        # update indices--boolean tensor of shape (B, max_length)
        # A position is updated if:
        # 1. The token changed (xt != new_xt)
        # 2. It's not a pad position
        # 3. It WAS a mask token that got unmasked (so we check xt == mask, not xt != mask)
        
        # Debug before fix
        old_update_ids = (xt != new_xt) & (xt != pad) & (xt != mask)
        
        # Correct logic: updated positions are where mask tokens were changed
        update_ids = (xt != new_xt) & (xt != pad)
        
        if self.insertion_planner is False:
            return new_xt, update_ids
        
        # ——— Poisson insertion (tau-leaping) — can insert multiple masks per gap ———
        ext = torch.poisson(len_rate * dt).long()  # (B, L+1)
        xt_len = xt.ne(pad).sum(dim=1)  # (B,)
        # Use ext.shape[1] to get the actual max_length dimension from the data
        actual_max_length = ext.shape[1] - 1  # ext is (B, L+1), so L = ext.shape[1] - 1
        gaps = torch.arange(ext.shape[1], device=device).view(1, -1)
        ext = ext * (gaps <= xt_len.view(batch_size, 1)).long()
        total_ext = ext.sum(dim=1)
        valid = xt_len + total_ext <= actual_max_length
        ext = ext * valid.view(batch_size, 1).long()

        ext_ex = ext.int().cumsum(dim=1)  # (B, L+1)
        new_len = xt_len + total_ext  # (B,)

        xt_tmp = torch.full_like(xt, pad)
        # Create position indices that match xt_tmp's shape
        pos_idx_for_fill = torch.arange(xt_tmp.shape[1], device=device).view(1, -1).expand(batch_size, -1)
        mask_fill = pos_idx_for_fill < new_len.view(batch_size, 1)
        xt_tmp[mask_fill] = mask

        new_pos_orig = pos_idx_L + ext_ex[:, :actual_max_length]  # (B, L)
        orig_mask = pos_idx_L < xt_len.view(batch_size, 1)
        flat_b = batch_idx_L[orig_mask]
        flat_p = new_pos_orig[orig_mask]
        xt_tmp[flat_b, flat_p] = new_xt[orig_mask]
        
        new_ins_xt = xt_tmp
        
        # Newly inserted masks: positions that are mask now but weren't before.
        newly_inserted_masks = (new_ins_xt == mask) & (xt != mask) & (xt != pad)
        
        update_ins_ids = newly_inserted_masks
        
        return new_xt, update_ids, new_ins_xt, update_ins_ids
    
    def loss_planner_flexible(self, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, eps=1e-3, centering=False, centering_strength=1.0, softmax_temperature=1.0):
        r"""
        Weighted denoising cross entropy loss
        X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
        
        log_rnd: [B] — pre-computed importance weights (already softmax-normalized over the full buffer)
        x: [B, L] (no mask)
        num_replicates: R, number of replicates of each row in x
        weight_func: w(lambda) for each sample, 1/lambda by default
        centering_strength: float, controls how much of the mean is subtracted (DMPO-style)
        softmax_temperature: float, temperature for softmax on log_rnd (>1 smooths weights)
        """
        
        batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, L]
        batch_size = batch.shape[0]
        
        batch_weights = (log_rnd.detach() / softmax_temperature).softmax(dim=-1)  # [B]
        if centering:
            batch_weights = batch_weights - centering_strength * batch_weights.mean()
        
        batch_weights = batch_weights.repeat_interleave(num_replicates, dim=0)
        
        lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
        lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
        
        t = lamda
        scale_factor = self.config.interpolant.max_length
        
        # compute unmasking and insertion loss
        interpolant_sample = self.interpolant.sample_interpolant(t, batch)
        unmask_weight, insert_weight = self.interpolant.elbo_weight(t, batch)

        prediction: ModelPrediction = self(interpolant_sample.xt, t)
        
        with torch.no_grad(): # no need to compute gradient in this step
            sampler_out = self.one_step_sampler(interpolant_sample.xt, t, prediction)
            # one_step_sampler returns (xs, update_ids) or (xs, update_ids, new_ins_xt, update_ins_ids)
            xs, update_ids = sampler_out[0], sampler_out[1]

        # The remasking head scores the freshly-decoded tokens to decide which to
        # remask, so it reads the POST-unmask state xs (matching inference, which
        # calls the planner on the decoded new_xt).
        planner = self.planner(xs, t)
        remasking_conf = planner["remasking_conf"]  # [B*R, L, 1]

        # Compute per-sample loss
        # IMPORTANT: interpolant_sample.xt has been reordered via st permutation
        # We need to map back to the original positions to compare with batch
        st = interpolant_sample.st  # [B*R, L] permutation indices
        batch_reordered = torch.gather(batch, 1, st)  # Apply same permutation to ground truth
        
        binary_label = (xs == batch_reordered).float() 
        
        # Only compute loss on positions that were updated
        per_token_loss = F.binary_cross_entropy_with_logits(
            remasking_conf.squeeze(-1),  # [B*R, L]
            binary_label,  # [B*R, L]
            reduction="none"  # [B*R, L]
        )
        
        per_token_loss = per_token_loss * update_ids.float()  # [B*R, L]
        
        # Mask out non-updated positions and average per sample
        per_sample_loss = per_token_loss.sum(dim=1) / (update_ids.sum(dim=1).float() + 1e-8)  # [B*R]
        
        # Weight by importance sampling weights
        weighted_loss = per_sample_loss * batch_weights  # [B*R]

        # ——— AUC / label-balance diagnostics (see loss_insert_planner_flexible) ———
        with torch.no_grad():
            metrics = {}
            sel_u = update_ids.bool()
            if sel_u.any():
                u_scores = remasking_conf.squeeze(-1)[sel_u]
                u_labels = binary_label[sel_u]
                metrics["unmask_auc"] = _binary_auc(u_scores, u_labels)
                metrics["unmask_label_mean"] = u_labels.mean().item()
                metrics["unmask_conf_mean"] = torch.sigmoid(u_scores).mean().item()
                metrics["unmask_n"] = float(sel_u.sum().item())
            self._last_planner_metrics = metrics

        return weighted_loss.mean()
    
    def loss_insert_planner_flexible(self, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, eps=1e-3, centering=False, centering_strength=1.0, softmax_temperature=1.0):
        r"""
        Weighted denoising cross entropy loss
        X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
        
        log_rnd: [B] — pre-computed importance weights
        x: [B, L] (no mask)
        num_replicates: R, number of replicates of each row in x
        weight_func: w(lambda) for each sample, 1/lambda by default
        centering_strength: float, controls how much of the mean is subtracted (DMPO-style)
        softmax_temperature: float, temperature for softmax on log_rnd (>1 smooths weights)
        """
        
        batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, L]
        batch_size = batch.shape[0]
        
        batch_weights = (log_rnd.detach() / softmax_temperature).softmax(dim=-1)  # [B]
        if centering:
            batch_weights = batch_weights - centering_strength * batch_weights.mean()
        
        batch_weights = batch_weights.repeat_interleave(num_replicates, dim=0)
        
        lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
        lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
        
        t = lamda
        scale_factor = self.config.interpolant.max_length
        
        # compute unmasking and insertion loss
        # deleted mask: binary tensor [B*R, L] where true tokens in batch were deleted
        # gap_assignment: [B*R, max_gaps, L] maps x1 positions to gap indices
        interpolant_sample, deleted_mask, gap_assignment = self.interpolant.sample_interpolant_plan(t, batch)
        unmask_weight, insert_weight = self.interpolant.elbo_weight(t, batch)

        prediction: ModelPrediction = self(interpolant_sample.xt, t)
        
        with torch.no_grad(): # no need to compute gradient in this step
            xs_unmask, update_unmask_ids, xs_insert, update_ins_ids = self.one_step_sampler(interpolant_sample.xt, t, prediction)

        # The remasking head scores the freshly-decoded tokens to decide which to
        # remask, so it must see the POST-unmask state xs_unmask (matching
        # inference in inference_quality.py, which calls the planner on the
        # decoded new_xt). Grad stays on here since this head is what we train.
        planner = self.planner(xs_unmask, t)
        remasking_conf = planner["remasking_conf"]  # [B*R, L, 1]

        # The insertion-quality head scores the freshly-inserted mask tokens, so
        # it must see the POST-insertion state xs_insert (aligned with
        # update_ins_ids / insertion_quality below, and matching inference in
        # remasking_scheduleaware.apply_schedule_aware_insertion). Grad stays on
        # here since this head is what we are training.
        if self.planner.insertion_planner:
            insertion_conf = self.planner(xs_insert, t)["insertion_conf"]  # [B*R, L, 1]
        else:
            insertion_conf = None
        
        # Compute per-sample loss
        # IMPORTANT: interpolant_sample.xt has been reordered via st permutation
        # We need to map back to the original positions to compare with batch
        # Use the st (permutation) to get the ground truth in the reordered space
        st = interpolant_sample.st  # [B*R, L] permutation indices
        batch_reordered = torch.gather(batch, 1, st)  # Apply same permutation to ground truth
        
        # Now compare in the reordered space
        binary_label = (xs_unmask == batch_reordered).float() 
        
        # Only compute loss on positions that were updated
        per_token_loss = F.binary_cross_entropy_with_logits(
            remasking_conf.squeeze(-1),  # [B*R, L]
            binary_label,  # [B*R, L]
            reduction="none"  # [B*R, L]
        )
        
        per_token_loss = per_token_loss * update_unmask_ids.float()  # [B*R, L]
        
        # Mask out non-updated positions and average per sample
        unmask_per_sample_loss = per_token_loss.sum(dim=1) / (update_unmask_ids.sum(dim=1).float() + 1e-8)  # [B*R]
        
        # compute insertion planner loss
        # For positions where masks were inserted, we evaluate the quality of insertion
        # by computing the probability that the ground truth token would be predicted at that position
        
        # IMPORTANT: We need to recompute predictions using xs_insert since that's where the masks were inserted
        # The original prediction was computed from xt (before insertion)
        with torch.no_grad():
            prediction_after_insert: ModelPrediction = self(xs_insert, t)
        
        # Get the token prediction probabilities at inserted mask positions
        # prediction_after_insert.token_logits: [B*R, L, V] - logits for all positions in xs_insert
        token_probs = F.softmax(prediction_after_insert.token_logits, dim=-1)  # [B*R, L, V]
        
        # For each gap where masks were inserted, compute the sum of probabilities
        # of the ground truth tokens that were deleted in that specific gap
        # gap_assignment: [B*R, max_gaps, L] - maps x1 positions to gap indices
        # batch: [B*R, L] - ground truth tokens in original space (before permutation)
        
        vocab_size = token_probs.shape[-1]
        L = token_probs.shape[1]
        max_gaps = gap_assignment.shape[1]
        
        # For each gap, create a vocabulary mask of tokens that belong to that gap
        # gap_vocab_mask[b, gap_idx, token_id] = 1 if token_id was deleted in gap gap_idx
        gap_vocab_mask = torch.zeros(batch_size, max_gaps, vocab_size, device=batch.device, dtype=torch.float)
        
        # Vectorized: gather tokens from batch for all gaps at once
        # tokens_expanded[b, gap_idx, pos] = batch[b, pos] for all positions
        tokens_expanded = batch.unsqueeze(1).expand(batch_size, max_gaps, L)  # [B*R, max_gaps, L]
        
        # valid_mask[b, gap_idx, pos] = 1 if position pos belongs to gap gap_idx and is not pad
        valid_mask = (gap_assignment > 0) & (tokens_expanded != self.interpolant.pad_token)  # [B*R, max_gaps, L]
        
        # Scatter tokens into vocabulary dimension: mark which tokens appear in each gap
        gap_vocab_mask.scatter_add_(
            2,  # scatter along vocabulary dimension
            tokens_expanded.clamp(0, vocab_size - 1),  # token indices [B*R, max_gaps, L]
            valid_mask.float()  # values to add [B*R, max_gaps, L]
        )
        
        # Binarize: a token either appears in the gap or not
        gap_vocab_mask = (gap_vocab_mask > 0).float()  # [B*R, max_gaps, V]
        
        # For each insertion position in xs_insert, determine which gap it corresponds to
        # Position p in xs_insert corresponds to gap p (insertions occur between existing tokens)
        # Vectorized: compute for all positions at once
        # token_probs: [B*R, L, V]
        # gap_vocab_mask[:, :L, :]: [B*R, L, V] - vocab mask for gaps 0 to L-1
        insertion_quality_full = (token_probs * gap_vocab_mask[:, :L, :]).sum(dim=-1)  # [B*R, L]
        
        # Only consider quality at positions where masks were actually inserted
        insertion_quality = insertion_quality_full * update_ins_ids.float()  # [B*R, L]
        
        # Compute insertion planner loss only if insertion_planner is enabled
        if insertion_conf is not None:
            # The planner predicts insertion confidence with insertion_conf
            # We want to train it to predict high confidence when insertion_quality is high
            # Use Bernoulli cross-entropy: treat insertion_quality as the "success probability"
            
            # Binary cross-entropy with insertion_quality as continuous labels in [0,1]
            ins_per_token_loss = F.binary_cross_entropy_with_logits(
                insertion_conf.squeeze(-1),  # [B*R, L] - planner's insertion confidence logits
                insertion_quality,  # [B*R, L] - ground truth token probability as quality metric
                reduction="none"
            )
            
            # Only compute loss where masks were actually inserted
            ins_per_token_loss = ins_per_token_loss * update_ins_ids.float()
            
            # Average per sample
            ins_per_sample_loss = ins_per_token_loss.sum(dim=1) / (update_ins_ids.sum(dim=1).float() + 1e-8)
        else:
            # No insertion planner - set loss to zero
            ins_per_sample_loss = torch.zeros_like(unmask_per_sample_loss)
        
        # Add to total loss
        per_sample_loss = unmask_per_sample_loss + ins_per_sample_loss
        
        # Weight by importance sampling weights
        weighted_loss = per_sample_loss * batch_weights  # [B*R]

        # ——— AUC / label-balance diagnostics (the loss alone hides degenerate
        # targets; near-0 BCE can mean "all labels one class", not "learned") ———
        with torch.no_grad():
            metrics = {}
            sel_u = update_unmask_ids.bool()
            if sel_u.any():
                u_scores = remasking_conf.squeeze(-1)[sel_u]
                u_labels = binary_label[sel_u]
                metrics["unmask_auc"] = _binary_auc(u_scores, u_labels)
                metrics["unmask_label_mean"] = u_labels.mean().item()
                metrics["unmask_conf_mean"] = torch.sigmoid(u_scores).mean().item()
                metrics["unmask_n"] = float(sel_u.sum().item())
            if insertion_conf is not None:
                sel_i = update_ins_ids.bool()
                if sel_i.any():
                    i_scores = insertion_conf.squeeze(-1)[sel_i]
                    i_targets = insertion_quality[sel_i]
                    i_labels = (i_targets > 0.5).float()
                    metrics["insert_auc"] = _binary_auc(i_scores, i_labels)
                    metrics["insert_target_mean"] = i_targets.mean().item()
                    metrics["insert_conf_mean"] = torch.sigmoid(i_scores).mean().item()
                    metrics["insert_n"] = float(sel_i.sum().item())
            self._last_planner_metrics = metrics

        return unmask_per_sample_loss.mean(), ins_per_sample_loss.mean(), weighted_loss.mean()
