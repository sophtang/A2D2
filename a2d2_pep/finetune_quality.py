# Distributed Data Parallel (DDP) finetuning for peptide generation using PyTorch Lightning
import argparse
import math
from datetime import datetime
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import os
import sys
from tqdm import tqdm
import pandas as pd

# add repo root (A2D2/) to sys.path so top-level packages like lightning_modules resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_quality import sample_peptides_buffer, sample_peptides_eval
from pep_utils.analyzer import PeptideAnalyzer
from pep_utils.utils import str2bool, set_seed
from pep_scoring.scoring_functions import ScoringFunctions
from pep_scoring.tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from lightning_modules.any_length_remask import AnyOrderInsertionFlowModuleFT
from lightning_modules import AnyOrderInsertionFlowModule
from tdc import Evaluator

# Repository root (two levels up from this file: A2D2/a2d2_pep/finetune_quality.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PeptideFinetuner(pl.LightningModule):
    """Lightning module for distributed peptide finetuning."""
    
    def __init__(
        self, 
        args, 
        policy_model, 
        reward_model, 
        tokenizer, 
        pretrained=None, 
        mcts=None,
        filename=None, 
        prot_name=None,
        eps=1e-5
    ):
        super().__init__()
        self.args = args
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.pretrained = pretrained
        self.mcts = mcts
        self.filename = filename
        self.prot_name = prot_name
        self.eps = eps
        
        # Length cutoff is tunable from the CLI: --min_peptide_bonds N enforces
        # >=N peptide bonds (filters degenerate short reward-hacked molecules);
        # --min_peptide_bonds 0 disables the cutoff.
        min_bonds = getattr(args, 'min_peptide_bonds', 4)
        self.analyzer = PeptideAnalyzer(
            min_peptide_bonds=max(0, min_bonds),
            enforce_min_peptide_bonds=min_bonds > 0,
        )

        # Save hyperparameters
        self.save_hyperparameters(ignore=['policy_model', 'reward_model', 'tokenizer', 'pretrained', 'mcts'])
        
        # Buffer for sequences
        self.x_saved = None
        self.log_rnd_saved = None
        self.final_rewards_saved = None
        
        # Logs
        self.valid_fraction_log = []
        self.uniqueness_log = []
        self.diversity_log = []
        self.affinity_log = []
        self.sol_log = []
        self.hemo_log = []
        self.nf_log = []
        self.permeability_log = []
        self._diversity_evaluator = Evaluator('diversity')
        
        # Alternating training between policy and planner
        self.train_policy = True  # Start by training policy
        self.alternation_frequency = getattr(args, 'alternation_frequency', 1)  # Alternate every N epochs
    
    def freeze_policy_model(self):
        """Freeze policy model parameters (but not planner)."""
        for name, param in self.policy_model.named_parameters():
            if not name.startswith('planner.'):
                param.requires_grad = False
    
    def unfreeze_policy_model(self):
        """Unfreeze policy model parameters (but not planner)."""
        for name, param in self.policy_model.named_parameters():
            if not name.startswith('planner.'):
                param.requires_grad = True
    
    def freeze_planner_model(self):
        """Freeze planner parameters."""
        if hasattr(self.policy_model, 'planner'):
            for param in self.policy_model.planner.parameters():
                param.requires_grad = False
    
    def unfreeze_planner_model(self):
        """Unfreeze planner parameters."""
        if hasattr(self.policy_model, 'planner'):
            for param in self.policy_model.planner.parameters():
                param.requires_grad = True
    
    def configure_optimizers(self):
        # Separate parameter groups for policy backbone vs planner heads
        planner_lr = getattr(self.args, 'planner_learning_rate', self.args.learning_rate)
        planner_params = []
        policy_params = []
        for name, param in self.policy_model.named_parameters():
            if name.startswith('planner.'):
                planner_params.append(param)
            else:
                policy_params.append(param)
        
        param_groups = [
            {'params': policy_params, 'lr': self.args.learning_rate},
            {'params': planner_params, 'lr': planner_lr},
        ]
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer
    
    def _get_quality_mode(self):
        """Map ablation flags + warmup state to quality_mode string."""
        if self.args.disable_planner:
            return "none"
        if self.current_epoch < self.args.schedule_warmup_epochs:
            return "none"
        di = getattr(self.args, 'disable_insertion_planner', False)
        du = getattr(self.args, 'disable_unmasking_planner', False)
        if di and du:
            return "none"
        if di:
            return "unmasking_only"
        if du:
            return "insertion_only"
        return "both"
    
    def on_save_checkpoint(self, checkpoint):
        """
        Save additional metadata to make loading easier.
        Saves the config directly in the checkpoint so loading doesn't need to follow references.
        """
        # Save the config from the policy model directly in the checkpoint
        if hasattr(self.policy_model, 'config'):
            checkpoint['config'] = self.policy_model.config
            print(f"Saved config to checkpoint for easier loading")
        
        # Save EMA params if they exist in the policy model
        if hasattr(self.policy_model, 'ema_params') and self.policy_model.ema_params:
            checkpoint['ema_params'] = self.policy_model.ema_params
            print(f"Saved EMA params to checkpoint")
        
        # Save planner state if it exists
        if hasattr(self.policy_model, 'planner'):
            checkpoint['planner_state'] = self.policy_model.planner.state_dict()
            print(f"Saved planner state to checkpoint")
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # If disable_planner mode, only train policy (no alternation)
        if self.args.disable_planner:
            self.train_policy = True
            self.unfreeze_policy_model()
            self.freeze_planner_model()
            if self.global_rank == 0 and self.current_epoch == 0:
                print(f"[FINETUNE_QUALITY] Training ONLY policy model (planner frozen, no remasking)")
        elif getattr(self.args, 'joint_training', False):
            # Joint mode: train policy + planner together every step (no alternation)
            self.train_policy = True  # marker; training_step adds planner loss when joint_training is set
            self.unfreeze_policy_model()
            self.unfreeze_planner_model()
            if self.global_rank == 0 and self.current_epoch == 0:
                print(f"[FINETUNE_QUALITY] JOINT TRAINING: policy + planner trained together (no alternation)")
        else:
            # Alternate between training policy and planner from epoch 0
            # Determine which model to train this epoch
            cycle_position = (self.current_epoch // self.alternation_frequency) % 2
            self.train_policy = (cycle_position == 0)
            
            if self.train_policy:
                # Train policy, freeze planner
                self.unfreeze_policy_model()
                self.freeze_planner_model()
                if self.global_rank == 0:
                    print(f"[ALTERNATION] Epoch {self.current_epoch}: Training POLICY model (planner frozen)")
            else:
                # Train planner, freeze policy
                self.freeze_policy_model()
                self.unfreeze_planner_model()
                if self.global_rank == 0:
                    print(f"[ALTERNATION] Epoch {self.current_epoch}: Training PLANNER model (policy frozen)")
        
        # Resample buffer if needed
        if self.x_saved is None or self.current_epoch % self.args.resample_every_n_step == 0:
            self._generate_buffer()
            # Synchronize all ranks after buffer generation to prevent NCCL timeout
            if self.trainer and self.trainer.world_size > 1:
                torch.distributed.barrier()
    
    def _generate_buffer(self):
        """Generate buffer of sequences for training - all ranks generate in parallel.

        When pool_size > 0, maintains a persistent pool and refreshes a fraction
        each time instead of regenerating the entire buffer from scratch. This
        preserves diversity/uniqueness across training by avoiding wholesale
        replacement with samples from an increasingly mode-collapsed policy.
        """
        world_size = self.trainer.world_size if self.trainer else 1
        rank = self.global_rank if self.trainer else 0

        pool_size = getattr(self.args, 'pool_size', 0)
        is_pool = pool_size > 0
        is_init = self.x_saved is None

        # Determine how many sequences to sample this call
        if is_pool:
            refresh_frac = getattr(self.args, 'pool_refresh_fraction', 0.2)
            if is_init:
                samples_per_gpu = pool_size
            else:
                samples_per_gpu = max(1, int(pool_size * refresh_frac))
            if rank == 0:
                if is_init:
                    print(f"\n[POOL] Initializing pool with {pool_size} sequences at epoch {self.current_epoch}")
                else:
                    print(f"\n[POOL] Refreshing {samples_per_gpu}/{pool_size} sequences ({refresh_frac*100:.0f}%) at epoch {self.current_epoch}")
        else:
            samples_per_gpu = self.args.buffer_size // world_size
            if rank == 0:
                samples_per_gpu += self.args.buffer_size % world_size

        accumulated_x = []
        accumulated_log_rnd = []
        accumulated_rewards = []
        total_accumulated = 0

        if rank == 0:
            print(f"\n[BUFFER] Starting buffer generation at epoch {self.current_epoch}")
            print(f"[BUFFER] GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"[BUFFER] GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            if not is_pool:
                print(f"[BUFFER] Each of {world_size} ranks will generate {samples_per_gpu} samples")
        
        max_attempts = getattr(self.args, 'max_buffer_attempts', 100)  # cap wasted GPU / infinite loop
        starvation_patience = getattr(self.args, 'buffer_starvation_patience', 10)
        attempts = 0

        import time
        while total_accumulated < samples_per_gpu and attempts < max_attempts:
            attempts += 1
            if rank == 0:
                print(f"[BUFFER] rank={rank} starting sampling attempt {attempts} at {time.strftime('%H:%M:%S')}")
            
            start_time = time.time()
            
            # new elbo loss
            if self.args.elbo_rnd:
                x_final, _, final_rewards, trace = \
                    sample_peptides_buffer(
                        self.policy_model,
                        self.reward_model, self.analyzer,
                        self.tokenizer,
                        steps=self.args.total_num_steps,
                        mask=self.policy_model.interpolant.mask_token,
                        pad=self.policy_model.interpolant.pad_token,
                        batch_size=self.args.batch_size,
                        max_length=self.args.max_length,
                        # Buffer generation never uses the quality heads (planner):
                        # the backbone must train on raw policy samples so that a
                        # poorly-trained planner can't corrupt the backbone's data.
                        quality_mode="none",
                        compute_rnd=False,
                        alpha=self.args.alpha,
                        num_remasking=self.args.num_remasking,
                        min_length=self.args.min_length,
                    )
                if x_final.shape[0] > 0:
                    with torch.no_grad():
                        noised = self.policy_model.prepare_noised_sample(
                            x_final, num_samples=self.args.elbo_rnd_num_samples)
                        policy_loss = self.policy_model.compute_loss_from_noised(noised)
                        pretrained_loss = self.pretrained.compute_loss_from_noised(noised)
                        log_rnd = (pretrained_loss - policy_loss) + (final_rewards / self.args.alpha)
                else:
                    log_rnd = torch.empty((0,), dtype=torch.float32, device=x_final.device)
            else:
                x_final, log_rnd, final_rewards, trace = \
                    sample_peptides_buffer(
                        self.policy_model,
                        self.reward_model, self.analyzer,
                        self.tokenizer,
                        steps=self.args.total_num_steps,
                        mask=self.policy_model.interpolant.mask_token,
                        pad=self.policy_model.interpolant.pad_token,
                        batch_size=self.args.batch_size,
                        max_length=self.args.max_length,
                        # Buffer generation never uses the quality heads (planner):
                        # the backbone must train on raw policy samples so that a
                        # poorly-trained planner can't corrupt the backbone's data.
                        quality_mode="none",
                        compute_rnd=True,
                        pretrained=self.pretrained,
                        alpha=self.args.alpha,
                        num_remasking=self.args.num_remasking,
                        min_length=self.args.min_length,
                    )
            elapsed = time.time() - start_time
            if rank == 0:
                print(f"[BUFFER] rank={rank} sampling took {elapsed:.1f}s")
            
            n_valid = x_final.shape[0]
            if n_valid > 0:
                accumulated_x.append(x_final)
                accumulated_log_rnd.append(log_rnd)
                accumulated_rewards.append(final_rewards)
                total_accumulated += n_valid
            
            if rank == 0:
                print(f"[BUFFER] rank={rank} epoch={self.current_epoch} quality_mode=none (heads disabled for buffer gen) accumulated={total_accumulated} / {samples_per_gpu} (batch yielded {n_valid} valid) attempt={attempts}")

            # Starvation guard: if nothing valid comes through (e.g. the length
            # cutoff is too aggressive for a collapsed policy), stop grinding GPU
            # hours and fail fast with an actionable message.
            if attempts >= starvation_patience and total_accumulated == 0:
                if rank == 0:
                    print(f"[BUFFER STARVATION] 0 valid samples after {attempts} attempts "
                          f"(min_peptide_bonds={getattr(self.args, 'min_peptide_bonds', 4)}). "
                          f"Aborting refill early — lower --min_peptide_bonds or check the policy.")
                break

        if total_accumulated == 0:
            raise RuntimeError(f"[BUFFER ERROR] Rank {rank}: No valid sequences generated after {attempts} attempts. Check sampling function and reward model.")

        if total_accumulated < samples_per_gpu:
            print(f"[BUFFER WARNING] Rank {rank}: Only generated {total_accumulated}/{samples_per_gpu} sequences after {attempts} attempts")

        new_x = torch.cat(accumulated_x, dim=0)[:samples_per_gpu]
        new_log_rnd = torch.cat(accumulated_log_rnd, dim=0)[:samples_per_gpu]
        new_rewards = torch.cat(accumulated_rewards, dim=0)[:samples_per_gpu]

        del accumulated_x, accumulated_log_rnd, accumulated_rewards
        torch.cuda.empty_cache()

        # Pool mode (after init): replace a random subset of the existing pool.
        # Classic mode / pool init: overwrite the buffer.
        if is_pool and not is_init:
            actual_new = min(new_x.shape[0], self.x_saved.shape[0])
            indices = torch.randperm(self.x_saved.shape[0], device=self.x_saved.device)[:actual_new]
            self.x_saved[indices] = new_x[:actual_new]
            self.log_rnd_saved[indices] = new_log_rnd[:actual_new]
            self.final_rewards_saved[indices] = new_rewards[:actual_new]
            if rank == 0:
                print(f"[POOL] Replaced {actual_new}/{self.x_saved.shape[0]} sequences, reward mean={self.final_rewards_saved.mean():.4f}")
        else:
            self.x_saved = new_x
            self.log_rnd_saved = new_log_rnd
            self.final_rewards_saved = new_rewards

        # Sanity check: median length (non-pad tokens) of buffered peptides.
        if rank == 0:
            pad = self.policy_model.interpolant.pad_token
            token_lens = (self.x_saved != pad).sum(dim=1)
            print(f"[BUFFER] peptide token length: median={token_lens.median().item()} "
                  f"min={token_lens.min().item()} max={token_lens.max().item()} "
                  f"(n={token_lens.shape[0]})")

    def training_step(self, batch, batch_idx):
        """Training step - batch is ignored, we use saved buffer."""
        # Use mini-batch sampling from buffer to avoid OOM
        buffer_size = self.x_saved.shape[0]
        mini_batch_size = getattr(self.args, 'training_mini_batch_size', 6)
        
        # Randomly sample mini_batch_size sequences from buffer
        if buffer_size > mini_batch_size:
            indices = torch.randperm(buffer_size, device=self.x_saved.device)[:mini_batch_size]
            x_final = self.x_saved[indices]
            log_rnd = self.log_rnd_saved[indices]
        else:
            # If buffer is smaller than mini_batch_size, use all
            x_final = self.x_saved
            log_rnd = self.log_rnd_saved
        
        joint = getattr(self.args, 'joint_training', False)
        policy_loss = None
        planner_loss = None

        if self.train_policy:
            # Train policy with WDCE loss
            policy_loss = self.policy_model.loss_wdce_flexible(
                log_rnd,
                x_final,
                num_replicates=self.args.wdce_num_replicates,
                centering=self.args.centering,
                centering_strength=self.args.centering_strength
            )
            self.log('train/policy_loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if (not self.train_policy) or joint:
            # Train planner with appropriate loss based on ablation flags
            if self.args.disable_insertion_planner:
                # Ablation: only train unmasking/remasking planner (no insertion head)
                planner_loss = self.policy_model.loss_planner_flexible(
                    log_rnd,
                    x_final,
                    num_replicates=self.args.wdce_num_replicates,
                    centering=self.args.centering,
                    centering_strength=self.args.centering_strength
                )
                self.log('train/planner_unmask_loss', planner_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_insert_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_loss', planner_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            elif self.args.disable_unmasking_planner:
                # Ablation: only train insertion planner (no remasking head)
                unmask_loss, insert_loss, _ = self.policy_model.loss_insert_planner_flexible(
                    log_rnd,
                    x_final,
                    num_replicates=self.args.wdce_num_replicates,
                    centering=self.args.centering,
                    centering_strength=self.args.centering_strength
                )
                # Zero out the unmasking component - only backprop insertion loss
                planner_loss = insert_loss
                self.log('train/planner_unmask_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_insert_loss', insert_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_loss', planner_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            else:
                # Full planner: train both remasking + insertion
                unmask_loss, insert_loss, planner_loss = self.policy_model.loss_insert_planner_flexible(
                    log_rnd,
                    x_final,
                    num_replicates=self.args.wdce_num_replicates,
                    centering=self.args.centering,
                    centering_strength=self.args.centering_strength
                )
                self.log('train/planner_unmask_loss', unmask_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_insert_loss', insert_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_loss', planner_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)


        # Combine losses depending on mode
        if joint:
            loss = policy_loss + planner_loss
            mode_value = 0.5
        elif self.train_policy:
            loss = policy_loss
            mode_value = 0.0
        else:
            loss = planner_loss
            mode_value = 1.0

        # Log overall loss and mode
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/mode', mode_value, prog_bar=True, sync_dist=True)
        
        return loss
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch - only rank 0 evaluates."""
        # Only evaluate every N epochs to save time
        eval_frequency = getattr(self.args, 'eval_every_n_epochs', 5)
        is_last_epoch = (self.trainer and self.current_epoch == self.trainer.max_epochs - 1)
        if self.global_rank == 0 and (self.current_epoch % eval_frequency == 0 or is_last_epoch):
            # Sample eval batch with updated policy
            valid_seqs, affinity, sol, hemo, nf, permeability, valid_fraction = \
                sample_peptides_eval(
                    self.policy_model, self.reward_model, self.analyzer,
                    self.tokenizer,
                    steps=self.args.total_num_steps,
                    mask=self.policy_model.interpolant.mask_token,
                    pad=self.policy_model.interpolant.pad_token,
                    batch_size=50,
                    max_length=self.args.max_length,
                    quality_mode=self._get_quality_mode(),
                    num_remasking=self.args.num_remasking,
                    return_valid=True,
                )

            # Uniqueness (% of valid that are unique) and Diversity
            # (1 - mean pairwise Tanimoto on Morgan FPs of unique sequences),
            # matching evaluate_peptide_table.py / evaluate_mol_table.py.
            num_valid = len(valid_seqs)
            unique_seqs = list(set(valid_seqs))
            num_unique = len(unique_seqs)
            uniqueness = num_unique / num_valid * 100.0 if num_valid > 0 else 0.0
            diversity = self._diversity_evaluator(unique_seqs) if num_unique > 1 else 0.0

            # Append to logs
            self.affinity_log.append(affinity)
            self.sol_log.append(sol)
            self.hemo_log.append(hemo)
            self.nf_log.append(nf)
            self.permeability_log.append(permeability)
            self.valid_fraction_log.append(valid_fraction)
            self.uniqueness_log.append(uniqueness)
            self.diversity_log.append(diversity)
            
            # Compute reward stats
            mean_reward = self.final_rewards_saved.mean().item()
            min_reward = self.final_rewards_saved.min().item()
            max_reward = self.final_rewards_saved.max().item()
            median_reward = self.final_rewards_saved.median().item()

            # Log metrics
            self.log_dict({
                "eval/affinity": np.mean(affinity),
                "eval/sol": np.mean(sol),
                "eval/hemo": np.mean(hemo),
                "eval/nf": np.mean(nf),
                "eval/permeability": np.mean(permeability),
                "eval/valid_fraction": valid_fraction,
                "eval/uniqueness": uniqueness,
                "eval/diversity": diversity,
                "eval/mean_reward_search": mean_reward,
                "eval/min_reward_search": min_reward,
                "eval/max_reward_search": max_reward,
                "eval/median_reward_search": median_reward
            })
            
            print(f"epoch {self.current_epoch} | affinity {np.mean(affinity):.4f} | "
                  f"sol {np.mean(sol):.4f} | hemo {np.mean(hemo):.4f} | "
                  f"nf {np.mean(nf):.4f} | permeability {np.mean(permeability):.4f} | "
                  f"valid {valid_fraction:.4f} | uniq {uniqueness:.2f}% | div {diversity:.4f}")
    
    def on_fit_end(self):
        """Called at the end of training - save results."""
        if self.global_rank == 0:
            # Save logs and plot
            base_path = self.args.base_path
            plot_path = f'{base_path}/results/{self.args.run_name}'
            os.makedirs(plot_path, exist_ok=True)
            
            output_log_path = f'{plot_path}/log_{self.filename}.csv'
            save_logs_to_file(self.valid_fraction_log, self.affinity_log,
                              self.sol_log, self.hemo_log, self.nf_log,
                              self.permeability_log, output_log_path,
                              uniqueness_log=self.uniqueness_log,
                              diversity_log=self.diversity_log)

            # Final generation
            x_eval, affinity, sol, hemo, nf, permeability, valid_fraction, df = \
                sample_peptides_eval(
                    self.policy_model, self.reward_model, self.analyzer,
                    self.tokenizer,
                    steps=self.args.total_num_steps, 
                    mask=self.policy_model.interpolant.mask_token, 
                    pad=self.policy_model.interpolant.pad_token, 
                    batch_size=50, 
                    max_length=self.args.max_length,
                    quality_mode=self._get_quality_mode(),
                    num_remasking=self.args.num_remasking,
                    dataframe=True,
                )
            df.to_csv(f'{plot_path}/{self.prot_name}_generation_results.csv', index=False)


def save_logs_to_file(valid_fraction_log, affinity_log,
                      sol_log, hemo_log, nf_log,
                      permeability_log, output_path,
                      uniqueness_log=None, diversity_log=None):
    """
    Saves the logs to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log_data = {
        "Iteration": list(range(1, len(valid_fraction_log) + 1)),
        "Valid Fraction": valid_fraction_log,
        "Binding Affinity": affinity_log,
        "Solubility": sol_log,
        "Hemolysis": hemo_log,
        "Nonfouling": nf_log,
        "Permeability": permeability_log,
    }
    if uniqueness_log is not None:
        log_data["Uniqueness (%)"] = uniqueness_log
    if diversity_log is not None:
        log_data["Diversity"] = diversity_log

    df = pd.DataFrame(log_data)
    df.to_csv(output_path, index=False)


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for Lightning trainer (we use buffer instead)."""
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return torch.zeros(1)  # Dummy data


def main():
    """Main entry point for distributed training."""
    # Disable DDP optimizer for higher-order ops like flex_attention
    import torch._dynamo
    torch._dynamo.config.optimize_ddp = False
    
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--base_path', type=str, default=REPO_ROOT)
    argparser.add_argument('--learning_rate', type=float, default=1e-4)
    argparser.add_argument('--num_epochs', type=int, default=100)
    argparser.add_argument('--num_accum_steps', type=int, default=4)
    argparser.add_argument('--truncate_steps', type=int, default=50)
    argparser.add_argument("--truncate_kl", type=str2bool, default=False)
    argparser.add_argument('--gumbel_temp', type=float, default=1.0)
    argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
    argparser.add_argument('--batch_size', type=int, default=50)
    argparser.add_argument('--name', type=str, default='debug')
    argparser.add_argument('--total_num_steps', type=int, default=128)
    argparser.add_argument('--copy_flag_temp', type=float, default=None)
    argparser.add_argument('--save_every_n_epochs', type=int, default=10)
    argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
    argparser.add_argument("--seed", type=int, default=0)
    # new
    argparser.add_argument('--run_name', type=str, default='peptides')
    argparser.add_argument("--save_path_dir", default=os.path.join(REPO_ROOT, "checkpoints", "finetune_peptides"), type=str)
    # mcts
    argparser.add_argument('--num_sequences', type=int, default=10)
    argparser.add_argument('--max_length', type=int, default=1024)
    argparser.add_argument('--min_length', type=int, default=0,
                           help='Minimum sequence length (in SMILES SPE tokens). '
                                'Samples shorter than this are dropped from the buffer. 0 disables the filter.')
    argparser.add_argument('--num_children', type=int, default=50)
    argparser.add_argument('--num_iter', type=int, default=30) 
    argparser.add_argument('--seq_length', type=int, default=1024)
    argparser.add_argument('--time_conditioning', action='store_true', default=False)
    argparser.add_argument('--mcts_sampling', type=int, default=0) # for batched categorical sampling: '0' means gumbel noise
    argparser.add_argument('--buffer_size', type=int, default=100)
    argparser.add_argument('--wdce_num_replicates', type=int, default=16)
    argparser.add_argument('--noise_removal', action='store_true', default=False)
    argparser.add_argument('--grad_clip', action='store_true', default=False)
    argparser.add_argument('--resample_every_n_step', type=int, default=10)
    argparser.add_argument('--exploration', type=float, default=0.1)
    argparser.add_argument('--reset_every_n_step', type=int, default=100)
    argparser.add_argument('--alpha', type=float, default=0.01)
    argparser.add_argument('--scalarization', type=str, default='sum')
    argparser.add_argument('--no_mcts', action='store_true', default=False)
    argparser.add_argument("--centering", action='store_true', default=False)
    argparser.add_argument("--centering_strength", type=float, default=1.0)
    
    # ELBO-based log_rnd estimation
    argparser.add_argument('--elbo_rnd', action='store_true', default=False,
        help='If set, compute log_rnd via ELBO instead of trajectory rollout')
    argparser.add_argument('--elbo_rnd_num_samples', type=int, default=16,
        help='Number of noisy time samples per sequence for ELBO-based log_rnd estimation')
    
    # adaptive schedule parameters
    argparser.add_argument('--use_adaptive_schedule', action='store_true', default=True)
    argparser.add_argument('--schedule_hidden_dim', type=int, default=256)
    argparser.add_argument('--schedule_num_layers', type=int, default=2)
    argparser.add_argument('--schedule_loss_weight', type=float, default=0.1)
    argparser.add_argument('--adaptive_threshold', type=float, default=0.5)
    argparser.add_argument('--freeze_base_model', action='store_true', default=False)
    argparser.add_argument('--schedule_warmup_epochs', type=int, default=0, help='Number of initial epochs to train WITHOUT remasking in buffer generation')
    argparser.add_argument('--alternation_frequency', type=int, default=20, help='Number of epochs to train each model before alternating (1=alternate every epoch)')
    argparser.add_argument('--planner_learning_rate', type=float, default=None, help='Separate learning rate for planner heads (defaults to --learning_rate if not set)')

    # objectives
    argparser.add_argument('--num_obj', type=int, default=5)
    argparser.add_argument('--prot_seq', type=str, default=None)
    argparser.add_argument('--prot_name', type=str, default='glast',
                           help='Protein target name. Looked up in PROTEINS dict unless --prot_seq is given.')
    argparser.add_argument('--devices', type=int, default=-1)
    argparser.add_argument('--checkpoint_path', type=str, default=None)
    argparser.add_argument('--resume_ckpt', type=str, default=None,
        help='Path to a Lightning last.ckpt to resume training from (restores epoch/optimizer/planner state). '
             'New checkpoints continue in the same directory as this checkpoint.')

    # remasking
    argparser.add_argument('--num_remasking', type=int, default=5)
    argparser.add_argument('--quality_threshold', type=float, default=1)

    # length cutoff (peptide-bond filter) + buffer starvation guard
    argparser.add_argument('--min_peptide_bonds', type=int, default=4,
        help='Minimum backbone peptide bonds for a sample to count as valid. '
             '0 disables the cutoff. Filters degenerate short reward-hacked molecules.')
    argparser.add_argument('--max_buffer_attempts', type=int, default=100,
        help='Max sampling rounds per buffer refill before giving up (caps wasted GPU when validity is low).')
    argparser.add_argument('--buffer_starvation_patience', type=int, default=10,
        help='If 0 valid samples after this many rounds, abort the refill early (starvation guard).')
    
    # planner ablation flags
    argparser.add_argument('--disable_planner', action='store_true', help='If set, disable remasking completely and only train policy (not planner) for quality optimization')
    argparser.add_argument('--disable_insertion_planner', action='store_true', help='Ablation: disable insertion quality filtering but keep unmasking/remasking planner')
    argparser.add_argument('--disable_unmasking_planner', action='store_true', help='Ablation: disable unmasking/remasking planner but keep insertion quality filtering')
    argparser.add_argument('--joint_training', action='store_true', help='Ablation: train policy and planner jointly each step (no alternation, both unfrozen, summed loss). Incompatible with --disable_planner.')
    
    # performance optimization
    argparser.add_argument('--eval_every_n_epochs', type=int, default=5, help='Evaluate only every N epochs to save time')
    argparser.add_argument('--num_training_steps_per_epoch', type=int, default=10, help='Number of gradient updates per epoch')
    argparser.add_argument('--training_mini_batch_size', type=int, default=6, help='Mini-batch size for training from buffer to avoid OOM')
    argparser.add_argument('--pool_size', type=int, default=0,
        help='If >0, maintain a persistent pool of this size and refresh a fraction each resample step (0=disabled, classic buffer). Helps preserve uniqueness/diversity over training.')
    argparser.add_argument('--pool_refresh_fraction', type=float, default=0.2,
        help='Fraction of pool to replace each resample step (only used when pool_size>0)')

    args = argparser.parse_args()
    
    # Default planner LR to policy LR if not specified
    if args.planner_learning_rate is None:
        args.planner_learning_rate = args.learning_rate
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Load models
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else \
        os.path.join(REPO_ROOT, 'pretrained', 'anylength_pep.ckpt')
    
    # Update args.checkpoint_path to ensure it's saved in hyperparameters for later inference
    args.checkpoint_path = checkpoint_path
    
    PROTEINS = {
        'amhr': 'MLGSLGLWALLPTAVEAPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLPPPGSPGTPGSQGPQAAPGESIWMALVLLGLFLLLLLLLGSIILALLQRKNYRVRGEPVPEPRPDSGRDWSVELQELPELCFSQVIREGGHAVVWAGQLQGKLVAIKAFPPRSVAQFQAERALYELPGLQHDHIVRFITASRGGPGRLLSGPLLVLELHPKGSLCHYLTQYTSDWGSSLRMALSLAQGLAFLHEERWQNGQYKPGIAHRDLSSQNVLIREDGSCAIGDLGLALVLPGLTQPPAWTPTQPQGPAAIMEAGTQRYMAPELLDKTLDLQDWGMALRRADIYSLALLLWEILSRCPDLRPDSSPPPFQLAYEAELGNTPTSDELWALAVQERRRPYIPSTWRCFATDPDGLRELLEDCWDADPEARLTAECVQQRLAALAHPQESHPFPESCPRGCPPLCPEDCTSIPAPTILPCRPQRSACHFSVQQGPCSRNPQPACTLSPV',
        'tfr':  'MMDQARSAFSNLFGGEPLSYTRFSLARQVDGDNSHVEMKLAVDEEENADNNTKANVTKPKRCSGSICYGTIAVIVFFLIGFMIGYLGYCKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPREAGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSKAATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVNAELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTDSTCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLKLAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKVSASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYPYLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADIKEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHVFWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF',
        'gfap': 'MERRRITSAARRSYVSSGEMMVGGLAPGRRLGPGTRLSLARMPPPLPTRVDFSLAGALNAGFKETRASERAEMMELNDRFASYIEKVRFLEQQNKALAAELNQLRAKEPTKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEATLARLDLERKIESLEEEIRFLRKIHEEEVRELQEQLARQQVHVELDVAKPDLTAALKEIRTQYEAMASSNMHEAEEWYRSKFADLTDAAARNAELLRQAKHEANDYRRQLQSLTCDLESLRGTNESLERQMREQEERHVREAASYQEALARLEEEGQSLKDEMARHLQEYQDLLNVKLALDIEIATYRKLLEGEENRITIPVQTFSNLQIRETSLDTKSVSEGHLKRNIVVKTVEMRDGEVIKESKQEHKDVM',
        'glp1': 'MAGAPGPLRLALLLLGMVGRAGPRPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERSSPEEQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQWDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVLSEQWIFRLYVSIGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVICIVVSKLKANLMCKTDIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQGLMVAILYCFVNNEVQLEFRKSWERWRLEHLHIQRDSSMKPLKCPTSSLSSGATAGSSMYTATCQASCS',
        'glast': 'MTKSNGEEPKMGGRMERFQQGVRKRTLLAKKKVQNITKEDVKSYLFRNAFVLLTVTAVIVGTILGFTLRPYRMSYREVKYFSFPGELLMRMLQMLVLPLIISSLVTGMAALDSKASGKMGMRAVVYYMTTTIIAVVIGIIIVIIIHPGKGTKENMHREGKIVRVTAADAFLDLIRNMFPPNLVEACFKQFKTNYEKRSFKVPIQANETLVGAVINNVSEAMETLTRITEELVPVPGSVNGVNALGLVVFSMCFGFVIGNMKEQGQALREFFDSLNEAIMRLVAVIMWYAPVGILFLIAGKIVEMEDMGVIGGQLAMYTVTVIVGLLIHAVIVLPLLYFLVTRKNPWVFIGGLLQALITALGTSSSSATLPITFKCLEENNGVDKRVTRFVLPVGATINMDGTALYEALAAIFIAQVNNFELNFGQIITISITATAASIGAAGIPQAGLVTMVIVLTSVGLPTDDITLIIAVDWFLDRLRTTTNVLGDSLGAGIVEHLSRHELKNRDVEMGNSVIEENEMKKPYQLIAQDNETEKPIDSETKM',
        'ncam': 'LQTKDLIWTLFFLGTAVSLQVDIVPSQGEISVGESKFFLCQVAGDAKDKDISWFSPNGEKLTPNQQRISVVWNDDSSSTLTIYNANIDDAGIYKCVVTGEDGSESEATVNVKIFQKLMFKNAPTPQEFREGEDAVIVCDVVSSLPPTIIWKHKGRDVILKKDVRFIVLSNNYLQIRGIKKTDEGTYRCEGRILARGEINFKDIQVIVNVPPTIQARQNIVNATANLGQSVTLVCDAEGFPEPTMSWTKDGEQIEQEEDDEKYIFSDDSSQLTIKKVDKNDEAEYICIAENKAGEQDATIHLKVFAKPKITYVENQTAMELEEQVTLTCEASGDPIPSITWRTSTRNISSEEKASWTRPEKQETLDGHMVVRSHARVSSLTLKSIQYTDAGEYICTASNTIGQDSQSMYLEVQYAPKLQGPVAVYTWEGNQVNITCEVFAYPSATISWFRDGQLLPSSNYSNIKIYNTPSASYLEVTPDSENDFGNYNCTAVNRIGQESLEFILVQADTPSSPSIDQVEPYSSTAQVQFDEPEATGGVPILKYKAEWRAVGEEVWHSKWYDAKEASMEGIVTIVGLKPETTYAVRLAALNGKGLGEISAASEF',
        'cereblon': 'MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQVIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANLTSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL',
        'ligase': 'MASQPPEDTAESQASDELECKICYNRYNLKQRKPKVLECCHRVCAKCLYKIIDFGDSPQGVIVCPFCRFETCLPDDEVSSLPDDNNILVNLTCGGKGKKCLPENPTELLLTPKRLASLVSPSHTSSNCLVITIMEVQRESSPSLSSTPVVEFYRPASFDSVTTVSHNWTVWNCTSLLFQTSIRVLVWLLGLLYFSSLPLGIYLLVSKKVTLGVVFVSLVPSSLVILMVYGFCQCVCHEFLDCMAPPS',
        'skp2': 'MHRKHLQEIPDLSSNVATSFTWGWDSSKTSELLSGMGVSALEKEEPDSENIPQELLSNLGHPESPPRKRLKSKGSDKDFVIVRRPKLNRENFPGVSWDSLPDELLLGIFSCLCLPELLKVSGVCKRWYRLASDESLWQTLDLTGKNLHPDVTGRLLSQGVIAFRCPRSFMDQPLAEHFSPFRVQHMDLSNSVIEVSTLHGILSQCSKLQNLSLEGLRLSDPIVNTLAKNSNLVRLNLSGCSGFSEFALQTLLSSCSRLDELNLSWCFDFTEKHVQVAVAHVSETITQLNLSGYRKNLQKSDLSTLVRRCPNLVHLDLSDSVMLKNDCFQEFFQLNYLQHLSLSRCYDIIPETLLELGEIPTLKTLQVFGIVPDGTLQLLKEALPHLQINCSHFTTIARPTIGNKKNQEIWGIKCRLTLQKPSCL',
    }

    if args.prot_seq is not None:
        prot = args.prot_seq
        prot_name = args.prot_name
    else:
        prot_name = args.prot_name
        if prot_name not in PROTEINS:
            raise ValueError(f"Unknown protein: {prot_name}. Choose from: {list(PROTEINS.keys())}")
        prot = PROTEINS[prot_name]
    filename = prot_name

    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.no_mcts:
        args.run_name = f'{curr_time}_adaptive_{prot_name}_resample{args.resample_every_n_step}_no-mcts'
    else:
        args.run_name = f'{curr_time}_adaptive_{prot_name}_resample{args.resample_every_n_step}_buffer{args.buffer_size}_numiter{args.num_iter}_children{args.num_children}'

    # Append ablation tags to run name for easy identification
    if args.disable_planner:
        args.run_name += '_no_planner'
    if args.disable_insertion_planner:
        args.run_name += '_no_insertion_planner'
    if args.disable_unmasking_planner:
        args.run_name += '_no_unmasking_planner'
    if args.joint_training:
        if args.disable_planner:
            raise ValueError("--joint_training is incompatible with --disable_planner (no planner to train)")
        args.run_name += '_joint_training'

    # When resuming, continue writing checkpoints into the SAME directory as the
    # checkpoint we resume from (keeps model-{epoch}.ckpt contiguous) instead of
    # spawning a fresh timestamped run directory.
    if args.resume_ckpt:
        args.save_path = os.path.dirname(os.path.abspath(args.resume_ckpt))
        args.run_name = os.path.basename(args.save_path)
    else:
        args.save_path = os.path.join(args.save_path_dir, args.run_name)
    os.makedirs(args.save_path, exist_ok=True)
    set_seed(args.seed, use_cuda=False)  # Don't init CUDA before Lightning spawns DDP workers

    # Initialize the model
    print("Loading models..")
    
    # Load pretrained model for reference (frozen)
    pretrained = AnyOrderInsertionFlowModule.load_from_checkpoint(checkpoint_path,
                                                map_location='cpu',
                                                weights_only=False)
    pretrained.eval()
    for param in pretrained.parameters():
        param.requires_grad = False
    
    # Load checkpoint to extract config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'hyper_parameters' in checkpoint:
        config = checkpoint['hyper_parameters']['config']
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("Cannot find config in checkpoint")
    
    # Update config for adaptive schedule
    from omegaconf import OmegaConf
    if not OmegaConf.is_config(config):
        from omegaconf import DictConfig
        config = DictConfig(config)
    
    # Disable struct mode to allow adding new keys
    OmegaConf.set_struct(config, False)
    
    config.training.use_adaptive_schedule = args.use_adaptive_schedule
    config.training.schedule_hidden_dim = args.schedule_hidden_dim
    config.training.schedule_num_layers = args.schedule_num_layers
    config.training.schedule_loss_weight = args.schedule_loss_weight
    config.training.freeze_base_model = args.freeze_base_model
    config.training.schedule_warmup_epochs = args.schedule_warmup_epochs
    
    # Re-enable struct mode
    OmegaConf.set_struct(config, True)
    
    # Initialize policy model with adaptive schedule
    policy_model = AnyOrderInsertionFlowModuleFT(
        config=config,
        args=args,
        pretrained_checkpoint=checkpoint_path,
        insertion_planner=True,
    )

    # define mcts
    score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling', 'permeability']

    tokenizer = SMILES_SPE_Tokenizer(
        os.path.join(REPO_ROOT, 'a2d2_pep', 'pep_scoring', 'tokenizer', 'new_vocab.txt'),
        os.path.join(REPO_ROOT, 'a2d2_pep', 'pep_scoring', 'tokenizer', 'new_splits.txt')
    )
    
    # Device will be set by Lightning automatically in DDP
    reward_model = ScoringFunctions(score_func_names, prot_seqs=[prot], device='cpu')
    model = PeptideFinetuner(
        args=args,
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        pretrained=pretrained,
        mcts=None,
        filename=filename,
        prot_name=prot_name
    )
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename='model-{epoch:02d}',
        every_n_epochs=args.save_every_n_epochs,
        save_top_k=-1,
        save_last=True,  # Also save last.ckpt
        auto_insert_metric_name=False
    )
    
    # Setup wandb logger - only on rank 0 to avoid multiple runs
    # Check if we're in a spawned DDP process
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if rank == 0:
        # Defaults to your default wandb entity; override with the WANDB_ENTITY env var.
        wandb_logger = WandbLogger(entity=os.environ.get('WANDB_ENTITY'), project='a2d2-pep', name=args.run_name)
    else:
        wandb_logger = None
    
    # Create dummy dataloader
    dataset = DummyDataset(size=args.num_training_steps_per_epoch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    # Setup trainer with DDP
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator='gpu',
        devices=args.devices,
        strategy=DDPStrategy(find_unused_parameters=True) if args.devices != 1 else 'auto',
        gradient_clip_val=args.gradnorm_clip if args.grad_clip else None,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=1
    )
    
    # Train (resume full training state from --resume_ckpt if provided).
    # weights_only=False is required when resuming because these checkpoints
    # store argparse.Namespace / OmegaConf objects in hyper_parameters, which
    # PyTorch 2.6's default weights_only=True unpickler rejects.
    if args.resume_ckpt:
        trainer.fit(model, dataloader, ckpt_path=args.resume_ckpt, weights_only=False)
    else:
        trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()