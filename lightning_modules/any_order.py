import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
import torch.nn.functional as F
from model.transformer import AnyOrderMaskInsertionFlow
from model.interpolant import AnyOrderMaskInsertionInterpolant, ModelPrediction
from .bregman import jump_kernel_elbo, mse
from .schedule import get_schedule_from_config


import re
from typing import Dict, Any


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


class AnyOrderInsertionFlowModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.model_type = config.interpolant.type
        self.learning_rate = config.training.learning_rate
        self.unmask_loss_fn = config.training.loss_fn.unmask
        self.insert_loss_fn = config.training.loss_fn.insert

        # Initialize model based on type
        self.model = AnyOrderMaskInsertionFlow(config)
        # self.model = torch.compile(self.model)  # Disabled: incompatible with flex_attention nested functions

        insert_schedule = get_schedule_from_config(config.interpolant.insert_schedule)
        unmask_schedule = get_schedule_from_config(config.interpolant.unmask_schedule)

        # Initialize interpolant
        self.interpolant = AnyOrderMaskInsertionInterpolant(
            insertion_schedule=insert_schedule,
            unmask_schedule=unmask_schedule,
            vocab_size=config.interpolant.tokens,
            mask_token=config.interpolant.mask_token,
            pad_token=config.interpolant.pad_token,
            max_length=config.interpolant.max_length,
        )

        # Save hyperparameters
        self.save_hyperparameters()

        self.ema_decay = config.training.ema_decay or 0.0
        self.use_ema = self.ema_decay > 0
        self._orig_params = {}

    def forward(self, x, t, return_features: bool = False):
        if self.config.training.only_embed_insert:
            result = self.model(x, self.interpolant.insertion_schedule.at(t), return_features=return_features)
        else:
            result = self.model(x, t, return_features=return_features)
        return result
    
    def get_hidden_states(self, indices: torch.Tensor, t: torch.Tensor):
        """Delegate to backbone transformer for RemaskingAnyOrder compatibility."""
        return self.model.get_hidden_states(indices, t)

    def training_loss(self, x1, t):
        interpolant_sample = self.interpolant.sample_interpolant(t, x1)
        unmask_weight, insert_weight = self.interpolant.elbo_weight(t, x1)

        prediction: ModelPrediction = self(interpolant_sample.xt, t)

        scale_factor = x1.shape[0] * self.config.interpolant.max_length

        match self.unmask_loss_fn:
            case "elbo":
                mask_indices = interpolant_sample.mask_indices
                unmask_loss = unmask_weight[mask_indices] * F.cross_entropy(
                    prediction.token_logits[mask_indices],
                    interpolant_sample.unmasked[mask_indices],
                    reduction="none",
                )
                unmask_loss = unmask_loss.sum() / scale_factor
            case _:
                raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_fn}")

        match self.insert_loss_fn:
            case "expectation":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss = insert_weight[gaps_mask] * jump_kernel_elbo(
                    gaps[gaps_mask], prediction.expected_gaps[gaps_mask]
                )
                insertion_loss = insertion_loss.sum() / scale_factor

            case "distribution":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss = insert_weight[gaps_mask] * F.cross_entropy(
                    prediction.length_posterior[gaps_mask], gaps[gaps_mask]
                )
                insertion_loss = insertion_loss.sum() / scale_factor

        total_loss = unmask_loss + insertion_loss
        return unmask_loss, insertion_loss, total_loss

    def prepare_noised_sample(self, x, num_samples=1, t=None):
        """
        Run the forward noising process on clean sequences x.
        Replicates each sequence num_samples times with independent random times
        so that both policy and pretrained can evaluate the same noised data.

        Args:
            x: [B, L] clean token sequences (no mask tokens)
            num_samples: K, number of noisy time samples per sequence
            t: [B*K] optional time values. If None, sampled uniformly.

        Returns:
            dict with all artifacts needed by compute_loss_from_noised.
        """
        B = x.shape[0]
        x_rep = x.repeat_interleave(num_samples, dim=0)  # [B*K, L]
        if t is None:
            t = torch.rand(B * num_samples, device=x.device)

        interpolant_sample = self.interpolant.sample_interpolant(t, x_rep)
        unmask_weight, insert_weight = self.interpolant.elbo_weight(t, x_rep)
        scale_factor = self.config.interpolant.max_length

        return {
            "interpolant_sample": interpolant_sample,
            "unmask_weight": unmask_weight,
            "insert_weight": insert_weight,
            "t": t,
            "scale_factor": scale_factor,
            "num_samples": num_samples,
            "batch_size": B,
        }

    def compute_loss_from_noised(self, noised):
        """
        Compute per-sample denoising loss given pre-noised data.
        Each model runs its own forward pass on the shared noised xt.

        Args:
            noised: dict from prepare_noised_sample()

        Returns:
            total_loss: [B] per-sample loss averaged over K noisy samples
        """
        interpolant_sample = noised["interpolant_sample"]
        unmask_weight = noised["unmask_weight"]
        insert_weight = noised["insert_weight"]
        t = noised["t"]
        scale_factor = noised["scale_factor"]
        num_samples = noised["num_samples"]
        B = noised["batch_size"]

        prediction: ModelPrediction = self(interpolant_sample.xt, t)

        match self.unmask_loss_fn:
            case "elbo":
                mask_indices = interpolant_sample.mask_indices
                unmask_loss_all = torch.zeros_like(unmask_weight)  # [B*K, L]
                unmask_loss_all[mask_indices] = unmask_weight[mask_indices] * F.cross_entropy(
                    prediction.token_logits[mask_indices],
                    interpolant_sample.unmasked[mask_indices],
                    reduction="none",
                )
                unmask_loss = unmask_loss_all.sum(dim=1) / scale_factor  # [B*K]
            case _:
                raise ValueError(f"Invalid unmask loss type: {self.unmask_loss_fn}")

        match self.insert_loss_fn:
            case "expectation":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss_all = torch.zeros_like(insert_weight)  # [B*K, L+1]
                insertion_loss_all[gaps_mask] = insert_weight[gaps_mask] * jump_kernel_elbo(
                    gaps[gaps_mask], prediction.expected_gaps[gaps_mask]
                )
                insertion_loss = insertion_loss_all.sum(dim=1) / scale_factor  # [B*K]
            case "distribution":
                gaps, gaps_mask = interpolant_sample.gaps_and_mask
                insertion_loss_all = torch.zeros_like(insert_weight)  # [B*K, L+1]
                insertion_loss_all[gaps_mask] = insert_weight[gaps_mask] * F.cross_entropy(
                    prediction.length_posterior[gaps_mask], gaps[gaps_mask]
                )
                insertion_loss = insertion_loss_all.sum(dim=1) / scale_factor  # [B*K]

        per_replicate_loss = unmask_loss + insertion_loss  # [B*K]
        per_sample_loss = per_replicate_loss.view(B, num_samples).mean(dim=1)  # [B]
        return per_sample_loss

    def loss_wdce_flexible(self, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, eps=1e-3, centering=False):
        r"""
        Weighted denoising cross entropy loss
        X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
        
        log_rnd: [B]; x: [B, L] (no mask)
        num_replicates: R, number of replicates of each row in x
        weight_func: w(lambda) for each sample, 1/lambda by default
        """
        
        print("logrnd shape:", log_rnd.shape)
        print("x shape:", x.shape)
        
        batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, L]
        
        batch_weights = log_rnd.detach().softmax(dim=-1) # [B*R]
        if centering:
            batch_weights = batch_weights - batch_weights.mean(dim=-1, keepdim=True)
        
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
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        eps = 1e-6
        interval = 1.0 - eps
        interval_size = interval / batch_size
        u = torch.rand(batch_size, device=device)
        return (torch.arange(batch_size, device=device, dtype=u.dtype) + u) * interval_size

    def training_step(self, batch, batch_idx):
        # Extract input data
        if isinstance(batch, dict):
            batch = batch["input_ids"]

        x1 = batch
        t = self.sample_time(x1.shape[0], x1.device)

        # Calculate the combined loss normally
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        warmup_steps = self.config.training.warmup_steps
        max_steps = self.config.training.max_steps

        # Always create a fresh schedule starting from step 0
        # This allows extending training beyond original max_steps
        linear_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
            last_epoch=-1,
        )
        post_warmup = max_steps - warmup_steps
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=post_warmup,
            eta_min=0.0,
            last_epoch=-1,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[linear_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
            last_epoch=-1,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
    ):
        super().optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure
        )
        # log learning rate and gradient norm
        lr = optimizer.param_groups[0]["lr"]
        self.log("train/lr", lr, on_step=True, prog_bar=True)
        grad_norm = torch.sqrt(
            sum(p.grad.norm(2) ** 2 for p in self.parameters() if p.grad is not None)
        )
        self.log("train/grad_norm", grad_norm, on_step=True, prog_bar=True)

        # update EMA
        if self.use_ema:
            for n, p in self.named_parameters():
                self.ema_params[n].mul_(self.ema_decay).add_(
                    p.data.clone().detach(), alpha=1 - self.ema_decay
                )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["config"] = self.config
        # save EMA state
        if self.use_ema:
            checkpoint["ema_params"] = {
                n: v.clone() for n, v in self.ema_params.items()
            }

    def on_load_checkpoint(self, checkpoint):
        self.config = checkpoint["config"]

        insert_schedule = get_schedule_from_config(
            self.config.interpolant.insert_schedule
        )
        unmask_schedule = get_schedule_from_config(
            self.config.interpolant.unmask_schedule
        )

        self.interpolant = AnyOrderMaskInsertionInterpolant(
            insertion_schedule=insert_schedule,
            unmask_schedule=unmask_schedule,
            vocab_size=self.config.interpolant.tokens,
            mask_token=self.config.interpolant.mask_token,
            pad_token=self.config.interpolant.pad_token,
            max_length=self.config.interpolant.max_length,
        )

        self.ema_params = checkpoint["ema_params"] if self.use_ema else {}

    def swap_to_ema(self):
        for name, p in self.named_parameters():
            self._orig_params[name] = p.data.clone()
            p.data.copy_(self.ema_params[name].to(p.device))

    def restore_original(self):
        for name, p in self.named_parameters():
            p.data.copy_(self._orig_params[name])
        self._orig_params.clear()

    def on_train_start(self):
        # initialize and move EMA buffers once model is on correct device
        if self.use_ema:
            self.ema_params = {
                name: param.clone().detach().to(self.device)
                for name, param in self.named_parameters()
            }
            for buf in self.ema_params.values():
                buf.requires_grad = False