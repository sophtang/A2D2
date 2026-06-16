import argparse
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

# imports
from inference_quality_mol import sample_mol_buffer, sample_mol_eval
from mol_utils.utils import str2bool, set_seed
from mol_scoring.scoring_functions import MolScoringFunctions
from lightning_modules.any_length_remask import AnyOrderInsertionFlowModuleFT
from lightning_modules import AnyOrderInsertionFlowModule
from safe.tokenizer import SAFETokenizer
from tdc import Evaluator

# Repository root (two levels up from this file: A2D2/a2d2_mol/finetune_mol.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_tokenizer():
    """Get SAFE tokenizer with added special tokens."""
    tk = SAFETokenizer.from_pretrained('datamol-io/safe-gpt').get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk

class MolFinetuner(pl.LightningModule):
    """Lightning module for distributed molecule finetuning."""
    
    def __init__(
        self, 
        args, 
        policy_model, 
        reward_model, 
        tokenizer, 
        pretrained=None, 
        mcts=None,
        filename=None, 
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
        self.eps = eps
        
        self.evaluator = Evaluator("diversity")
                
        # Save hyperparameters
        self.save_hyperparameters(ignore=['policy_model', 'reward_model', 'tokenizer', 'pretrained', 'mcts'])
        
        # Buffer for sequences
        self.x_saved = None
        self.log_rnd_saved = None
        self.final_rewards_saved = None
        
        # initialize logs
        self.valid_fraction_log = []
        self.diversity_log = []
        self.qed_log = []
        self.sa_log = []
        self.quality_log = []
        self.uniqueness_log = []
        
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
            if self.global_rank == 0:
                print(f"[BUFFER] Starting buffer generation for epoch {self.current_epoch}")
            self._generate_buffer()
            # Synchronize all ranks after buffer generation
            if self.trainer and self.trainer.world_size > 1:
                if self.global_rank == 0:
                    print(f"[BUFFER] All ranks completed buffer generation, synchronizing...")
                torch.distributed.barrier()
                if self.global_rank == 0:
                    print(f"[BUFFER] Synchronization complete!")
    
    def _generate_buffer(self):
        """Generate buffer of sequences for training.
        
        When pool_size > 0, maintains a persistent pool and refreshes a fraction
        each time instead of regenerating the entire buffer from scratch.
        """
        rank = self.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1
        
        pool_size = getattr(self.args, 'pool_size', 0)
        is_pool = pool_size > 0
        is_init = self.x_saved is None
        
        # Determine how many molecules to sample this call
        if is_pool:
            refresh_frac = getattr(self.args, 'pool_refresh_fraction', 0.2)
            if is_init:
                samples_per_gpu = pool_size
            else:
                samples_per_gpu = max(1, int(pool_size * refresh_frac))
            if rank == 0:
                if is_init:
                    print(f"\n[POOL] Initializing pool with {pool_size} molecules at epoch {self.current_epoch}")
                else:
                    print(f"\n[POOL] Refreshing {samples_per_gpu}/{pool_size} molecules ({refresh_frac*100:.0f}%) at epoch {self.current_epoch}")
        else:
            samples_per_gpu = self.args.buffer_size // world_size
            if rank == 0:
                samples_per_gpu += self.args.buffer_size % world_size
        
        if rank == 0:
            print(f"\n[BUFFER] Starting buffer generation at epoch {self.current_epoch}")
        
        accumulated_x = []
        accumulated_log_rnd = []
        accumulated_rewards = []
        total_accumulated = 0
        
        max_attempts = 100  # Prevent infinite loop
        attempts = 0
        
        import time
        while total_accumulated < samples_per_gpu and attempts < max_attempts:
            attempts += 1
            if rank == 0:
                print(f"[BUFFER] rank={rank} starting sampling attempt {attempts} at {time.strftime('%H:%M:%S')}")
            
            start_time = time.time()
            
            x_final, log_rnd, final_rewards, trace = \
                sample_mol_buffer(
                    self.policy_model,
                    self.pretrained,
                    self.reward_model,
                    self.tokenizer,
                    steps=self.args.total_num_steps,
                    mask=self.policy_model.interpolant.mask_token,
                    pad=self.policy_model.interpolant.pad_token,
                    batch_size=self.args.batch_size,
                    max_length=self.args.max_length,
                    quality_mode=self._get_quality_mode(),
                    alpha=self.args.alpha,
                    num_remasking=self.args.num_remasking,
                    quality_threshold=self.args.quality_threshold,
                    use_quality_filter=self.args.use_quality_filter,
                )
            if self.args.elbo_rnd:
                # Override trajectory log_rnd with forward ELBO estimate
                if x_final.shape[0] > 0:
                    with torch.no_grad():
                        noised = self.policy_model.prepare_noised_sample(
                            x_final, num_samples=self.args.elbo_rnd_num_samples)
                        policy_loss = self.policy_model.compute_loss_from_noised(noised)
                        pretrained_loss = self.pretrained.compute_loss_from_noised(noised)
                        log_rnd = (pretrained_loss - policy_loss) + (final_rewards / self.args.alpha)
            
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
                qm = self._get_quality_mode()
                print(f"[BUFFER] rank={rank} epoch={self.current_epoch} quality_mode={qm} accumulated={total_accumulated} / {samples_per_gpu} (batch yielded {n_valid} valid) attempt={attempts}")
        
        if total_accumulated == 0:
            raise RuntimeError(f"[BUFFER ERROR] Rank {rank}: No valid sequences generated after {attempts} attempts. Check sampling function and reward model.")
        
        if total_accumulated < samples_per_gpu:
            print(f"[BUFFER WARNING] Rank {rank}: Only generated {total_accumulated}/{samples_per_gpu} sequences after {attempts} attempts")
        
        new_x = torch.cat(accumulated_x, dim=0)[:samples_per_gpu]
        new_log_rnd = torch.cat(accumulated_log_rnd, dim=0)[:samples_per_gpu]
        new_rewards = torch.cat(accumulated_rewards, dim=0)[:samples_per_gpu]
        
        del accumulated_x, accumulated_log_rnd, accumulated_rewards
        torch.cuda.empty_cache()
        
        # add to buffer: pool mode replaces a random subset, classic mode overwrites
        if is_pool and not is_init:
            actual_new = min(new_x.shape[0], self.x_saved.shape[0])
            indices = torch.randperm(self.x_saved.shape[0], device=self.x_saved.device)[:actual_new]
            self.x_saved[indices] = new_x[:actual_new]
            self.log_rnd_saved[indices] = new_log_rnd[:actual_new]
            self.final_rewards_saved[indices] = new_rewards[:actual_new]
            if rank == 0:
                print(f"[POOL] Replaced {actual_new}/{self.x_saved.shape[0]} molecules, reward mean={self.final_rewards_saved.mean():.4f}")
        else:
            self.x_saved = new_x
            self.log_rnd_saved = new_log_rnd
            self.final_rewards_saved = new_rewards
        
        if rank == 0:
            print(f"[BUFFER] After cleanup - GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    def training_step(self, batch, batch_idx):
        """Training step - batch is ignored, we use saved buffer."""
        # Process buffer in mini-batches to avoid OOM
        mini_batch_size = getattr(self.args, 'training_mini_batch_size', 8)
        buffer_size = self.x_saved.shape[0]
        
        # Randomly sample a mini-batch from buffer
        indices = torch.randperm(buffer_size, device=self.x_saved.device)[:mini_batch_size]
        x_final = self.x_saved[indices]
        
        # get log_rnd values
        log_rnd = self.log_rnd_saved[indices]
        
        sm_temp = getattr(self.args, 'softmax_temperature', 1.0)
        
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
                centering_strength=self.args.centering_strength,
                softmax_temperature=sm_temp,
            )
            self.log('train/policy_loss', policy_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        if (not self.train_policy) or joint:
            # Train planner with appropriate loss based on ablation flags
            if self.args.disable_insertion_planner:
                # Ablation: only train unmasking planner (no insertion head)
                planner_loss = self.policy_model.loss_planner_flexible(
                    log_rnd,
                    x_final,
                    num_replicates=self.args.wdce_num_replicates,
                    centering=self.args.centering,
                    centering_strength=self.args.centering_strength,
                    softmax_temperature=sm_temp,
                )
                self.log('train/planner_unmask_loss', planner_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_insert_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log('train/planner_loss', planner_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            elif self.args.disable_unmasking_planner:
                # only train insertion planner (no remasking head)
                unmask_loss, insert_loss, _ = self.policy_model.loss_insert_planner_flexible(
                    log_rnd,
                    x_final,
                    num_replicates=self.args.wdce_num_replicates,
                    centering=self.args.centering,
                    centering_strength=self.args.centering_strength,
                    softmax_temperature=sm_temp,
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
                    centering_strength=self.args.centering_strength,
                    softmax_temperature=sm_temp,
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
            x_eval, qed, sa, uniqueness, diversity, quality, valid_fraction = \
                sample_mol_eval(
                    self.policy_model, self.reward_model,
                    self.tokenizer,
                    steps=self.args.total_num_steps, 
                    mask=self.policy_model.interpolant.mask_token, 
                    pad=self.policy_model.interpolant.pad_token, 
                    batch_size=50, 
                    max_length=self.args.max_length,
                    quality_mode=self._get_quality_mode(),
                    num_remasking=self.args.num_remasking,
                    evaluator=self.evaluator,
                )
            
            # Append to logs
            self.valid_fraction_log.append(valid_fraction)
            self.uniqueness_log.append(uniqueness)
            self.diversity_log.append(diversity)
            self.qed_log.append(qed)
            self.sa_log.append(sa)
            self.quality_log.append(quality)
            
            # Compute reward stats
            mean_reward = self.final_rewards_saved.mean().item()
            min_reward = self.final_rewards_saved.min().item()
            max_reward = self.final_rewards_saved.max().item()
            median_reward = self.final_rewards_saved.median().item()
            
            # Log metrics
            self.log_dict({
                "eval/valid_fraction": valid_fraction,
                "eval/uniqueness": np.mean(uniqueness),
                "eval/diversity": np.mean(diversity),
                "eval/qed": np.mean(qed),
                "eval/sa": np.mean(sa),
                "eval/quality": np.mean(quality),
                "eval/mean_reward_search": mean_reward,
                "eval/min_reward_search": min_reward,
                "eval/max_reward_search": max_reward,
                "eval/median_reward_search": median_reward
            })
            
            print(f"epoch {self.current_epoch} | validity {valid_fraction:.4f} | uniqueness {np.mean(uniqueness):.4f} | diversity {np.mean(diversity):.4f} | "
                  f"QED {np.mean(qed):.4f} | SA {np.mean(sa):.4f} | quality {np.mean(quality):.4f} | ")
    
    def on_fit_end(self):
        """Called at the end of training - save results."""
        if self.global_rank == 0:
            # Save logs and plot
            base_path = self.args.base_path
            plot_path = f'{base_path}/results/{self.args.run_name}'
            os.makedirs(plot_path, exist_ok=True)
            
            output_log_path = f'{plot_path}/log_{self.filename}.csv'
            save_logs_to_file(self.valid_fraction_log, self.uniqueness_log,
                              self.diversity_log, self.qed_log, self.sa_log,
                              self.quality_log, output_log_path)

            # Final generation
            x_eval, qed, sa, valid_fraction, uniqueness, diversity, quality, df = \
                sample_mol_eval(
                    self.policy_model, self.reward_model,
                    self.tokenizer,
                    steps=self.args.total_num_steps, 
                    mask=self.policy_model.interpolant.mask_token, 
                    pad=self.policy_model.interpolant.pad_token, 
                    batch_size=50, 
                    max_length=self.args.max_length,
                    quality_mode=self._get_quality_mode(),
                    num_remasking=self.args.num_remasking,
                    evaluator=self.evaluator,
                    dataframe=True,
                )
            df.to_csv(f'{plot_path}/mol_generation_results.csv', index=False)


def save_logs_to_file(valid_fraction_log, uniqueness_log, 
                      diversity_log, qed_log, sa_log, 
                      quality_log, output_path):
    """
    Saves the logs to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    log_data = {
        "Iteration": list(range(1, len(valid_fraction_log) + 1)),
        "Valid Fraction": valid_fraction_log,
        "Uniqueness": uniqueness_log,
        "Diversity": diversity_log,
        "QED": qed_log,
        "Synthetic Accessibility": sa_log,
        "Quality": quality_log
    }
    
    df = pd.DataFrame(log_data)
    df.to_csv(output_path, index=False)


class DummyDataset(torch.utils.data.Dataset):
    """Dummy dataset for Lightning trainer (we use buffer instead)."""
    def __init__(self, size=100):
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
    argparser.add_argument('--eval_every_n_epochs', type=int, default=5, help='Evaluate only every N epochs to save time')
    argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
    argparser.add_argument("--seed", type=int, default=0)
    # new
    argparser.add_argument('--run_name', type=str, default='mol')
    argparser.add_argument("--save_path_dir", default="", type=str)
    # mcts
    argparser.add_argument('--num_sequences', type=int, default=10)
    argparser.add_argument('--max_length', type=int, default=1024)
    argparser.add_argument('--num_children', type=int, default=50)
    argparser.add_argument('--num_iter', type=int, default=30) # iterations of mcts
    argparser.add_argument('--seq_length', type=int, default=1024)
    argparser.add_argument('--time_conditioning', action='store_true', default=False)
    argparser.add_argument('--mcts_sampling', type=int, default=0) # for batched categorical sampling: '0' means gumbel noise
    argparser.add_argument('--buffer_size', type=int, default=100)
    argparser.add_argument('--wdce_num_replicates', type=int, default=16)
    argparser.add_argument('--noise_removal', action='store_true', default=False)
    argparser.add_argument('--grad_clip', action='store_true', default=False)
    argparser.add_argument('--resample_every_n_step', type=int, default=3)
    argparser.add_argument('--exploration', type=float, default=0.1)
    argparser.add_argument('--reset_every_n_step', type=int, default=100)
    argparser.add_argument('--alpha', type=float, default=0.01)
    argparser.add_argument('--scalarization', type=str, default='sum')
    argparser.add_argument('--no_mcts', action='store_true', default=False)
    argparser.add_argument("--centering", action='store_true', default=False)
    argparser.add_argument("--centering_strength", type=float, default=1.0)
    
    # adaptive schedule parameters
    argparser.add_argument('--use_adaptive_schedule', action='store_true', default=True)
    argparser.add_argument('--schedule_hidden_dim', type=int, default=256)
    argparser.add_argument('--schedule_num_layers', type=int, default=2)
    argparser.add_argument('--schedule_loss_weight', type=float, default=0.1)
    argparser.add_argument('--adaptive_threshold', type=float, default=0.5)
    argparser.add_argument('--freeze_base_model', action='store_true', default=False)
    argparser.add_argument('--schedule_warmup_epochs', type=int, default=20, help='Number of initial epochs to train WITHOUT remasking in buffer generation')
    argparser.add_argument('--alternation_frequency', type=int, default=5, help='Number of epochs to train each model before alternating (1=alternate every epoch)')
    argparser.add_argument('--planner_learning_rate', type=float, default=None, help='Separate learning rate for planner heads (defaults to --learning_rate if not set)')

    # objectives
    argparser.add_argument('--num_obj', type=int, default=2)
    argparser.add_argument('--devices', type=int, default=-1)
    argparser.add_argument('--checkpoint_path', type=str, default=None)
    
    # ELBO-based log_rnd estimation
    argparser.add_argument('--elbo_rnd', action='store_true', default=False,
        help='If set, compute log_rnd via forward ELBO instead of trajectory rollout')
    argparser.add_argument('--elbo_rnd_num_samples', type=int, default=4,
        help='Number of noisy time samples per sequence for ELBO-based log_rnd estimation')

    # remasking 
    argparser.add_argument('--num_remasking', type=int, default=5)
    argparser.add_argument('--quality_threshold', type=float, default=1)
    argparser.add_argument('--use_quality_filter', action='store_true', help='If set, filter buffer to only include molecules with QED>=0.6 and SA<=4')
    argparser.add_argument('--training_mini_batch_size', type=int, default=8, help='Mini-batch size for training step to avoid OOM')
    argparser.add_argument('--disable_planner', action='store_true', help='If set, disable remasking completely and only train policy (not planner) for quality optimization')
    argparser.add_argument('--disable_insertion_planner', action='store_true', help='Ablation: disable insertion quality filtering but keep unmasking/remasking planner')
    argparser.add_argument('--disable_unmasking_planner', action='store_true', help='Ablation: disable unmasking/remasking planner but keep insertion quality filtering')
    argparser.add_argument('--joint_training', action='store_true', help='Ablation: train policy and planner jointly each step (no alternation, both unfrozen, summed loss). Incompatible with --disable_planner.')
    argparser.add_argument('--qed_only', action='store_true', help='If set, optimize only for QED score (no SA)')
    argparser.add_argument('--softmax_temperature', type=float, default=1.0,
        help='Temperature for softmax on importance weights (>1 smooths, prevents concentration)')
    argparser.add_argument('--pool_size', type=int, default=0,
        help='If >0, maintain a persistent pool of this size and refresh a fraction each resample step (0=disabled, classic buffer)')
    argparser.add_argument('--pool_refresh_fraction', type=float, default=0.2,
        help='Fraction of pool to replace each resample step (only used when pool_size>0)')
    argparser.add_argument('--num_training_steps_per_epoch', type=int, default=10,
        help='Number of gradient updates per epoch (1=original, 10=recommended)')

    args = argparser.parse_args()
    
    # Default planner LR to policy LR if not specified
    if args.planner_learning_rate is None:
        args.planner_learning_rate = args.learning_rate
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Load models
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else \
        os.path.join(REPO_ROOT, 'pretrained', 'anylength_mol.ckpt')
    
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.no_mcts:
        args.run_name = f'mol_al_resample{args.resample_every_n_step}_no-mcts_{curr_time}'
    else:
        args.run_name = f'mol_al_resample{args.resample_every_n_step}_buffer{args.buffer_size}_numiter{args.num_iter}_children{args.num_children}_{curr_time}'

    # append ablation tags to run name for easy identification
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
    
    OmegaConf.set_struct(config, False)
    
    config.training.use_adaptive_schedule = args.use_adaptive_schedule
    config.training.schedule_hidden_dim = args.schedule_hidden_dim
    config.training.schedule_num_layers = args.schedule_num_layers
    config.training.schedule_loss_weight = args.schedule_loss_weight
    config.training.freeze_base_model = args.freeze_base_model
    config.training.schedule_warmup_epochs = args.schedule_warmup_epochs
    config.training.use_bracket_safe = True
    
    OmegaConf.set_struct(config, True)
    
    # initialize policy model with adaptive schedule
    policy_model = AnyOrderInsertionFlowModuleFT(
        config=config,
        args=args,
        pretrained_checkpoint=checkpoint_path,
        insertion_planner=True,
    )

    # define mcts
    if args.qed_only:
        score_func_names = ['qed'] 
    else:
        score_func_names = ['qed', 'sa']  

    tokenizer = get_tokenizer()
    
    filename = args.run_name
    
    # Device will be set by Lightning automatically in DDP
    reward_model = MolScoringFunctions(score_func_names, device='cpu')
    model = MolFinetuner(
        args=args,
        policy_model=policy_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        pretrained=pretrained,
        mcts=None,
        filename=filename,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_path,
        filename='model-{epoch:02d}-{train_loss:.4f}',
        every_n_epochs=args.save_every_n_epochs,
        save_top_k=-1,  # Save all checkpoints
        save_last=True,  # Also save last.ckpt
        auto_insert_metric_name=False
    )
    
    # Defaults to your default wandb entity; override with the WANDB_ENTITY env var.
    wandb_logger = WandbLogger(entity=os.environ.get('WANDB_ENTITY'), project='a2d2-mol', name=args.run_name)
    
    # create dummy dataloader
    dataset = DummyDataset(size=args.num_training_steps_per_epoch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
    # setup trainer with DDP
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
    
    # Train
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()