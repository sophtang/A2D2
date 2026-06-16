import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
import argparse
import hydra
from omegaconf import OmegaConf
from datetime import datetime
# Directory containing this file and the config_*.yaml files (used by Hydra below).
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the repo root (A2D2/) to sys.path so top-level packages like lightning_modules resolve.
sys.path.insert(0, os.path.dirname(CONFIG_DIR))

import wandb
from lightning_modules import AnyOrderInsertionFlowModule


torch.set_printoptions(threshold=10_000)
torch.set_float32_matmul_precision("high")

# Disable DDP optimizer due to incompatibility with flex_attention higher-order ops
torch._dynamo.config.optimize_ddp = False

def train(config):
	wandb_logger = None

	# set the random seed
	pl.seed_everything(42)
	torch.manual_seed(42)
  
	# Only initialize wandb on rank 0 to avoid multiple runs
	if int(os.environ.get("LOCAL_RANK", 0)) == 0:
		wandb.init(
			project=config.wandb.project, 
			name=config.wandb.name, 
			config=OmegaConf.to_container(config, resolve=True),  # Convert to dict
			dir=config.wandb.path
		)
		wandb_logger = WandbLogger(
				project=wandb.run.project,
				name=wandb.run.name,
				log_model=False,  # Disable checkpoint uploading to save disk space
			)

	# Modify config to add timestamp to checkpoint directory
	OmegaConf.set_struct(config, False)
	time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
	config.training.checkpoint_dir = os.path.join(
		config.training.checkpoint_dir, time_string
	)
	OmegaConf.set_struct(config, True)

	# Create checkpoint directory
	os.makedirs(config.training.checkpoint_dir, exist_ok=True)
	
	# Setup data module - check if using HuggingFace dataset
	if hasattr(config, 'hf_dataset'):
		# Imported lazily: the HF/SAFE path is only used by the molecule configs,
		# which keep mol_dataset.py (and its `safe` dependency) in a2d2_mol/.
		from mol_dataset import setup_hf_data_and_update_config
		print(f"Using HuggingFace dataset: {config.hf_dataset.name}")
		data_module = setup_hf_data_and_update_config(
			config,
			dataset_name=config.hf_dataset.name,
			smiles_column=config.hf_dataset.get('smiles_column', 'smiles')
		)
	else:
		# Imported lazily: the local (arrow) path is used by the peptide config,
		# which keeps dataloading_for_dynamic_batching.py in a2d2_pep/.
		from data.dataloading_for_dynamic_batching import setup_data_and_update_config
		print("Using local dataset")
		data_module = setup_data_and_update_config(config)
	
	module = AnyOrderInsertionFlowModule(config)
 
	# Initialize trainer

	# Configure trainer arguments
	# Map torch_dtype to Lightning precision
	dtype_str = config.model.get('torch_dtype', 'bfloat16')
	precision_map = {
		'float32': '32-true',
		'float16': '16-mixed',
		'bfloat16': 'bf16-mixed'
	}
	precision = precision_map.get(dtype_str, 'bf16-mixed')
	
	trainer_kwargs = dict(
		num_nodes=config.training.nodes,
		accelerator="gpu",
		devices=config.training.devices,
		strategy="ddp",
		precision=precision,
		accumulate_grad_batches=(
			config.training.batch_size
			// (
				config.training.per_gpu_batch_size
				* config.training.nodes
				* config.training.devices
			)
		),
		log_every_n_steps=10,
		enable_checkpointing=True,
		default_root_dir=config.training.checkpoint_dir,
		gradient_clip_val=1.0,
	)
	# Only one of max_steps or max_epochs will be used
	if config.training.max_steps is not None:
		trainer_kwargs["max_steps"] = config.training.max_steps
	elif config.training.num_epochs is not None:
		trainer_kwargs["max_epochs"] = config.training.num_epochs
		config.training.max_steps = config.training.max_steps
	else:
		raise ValueError(
			"Either max_steps or num_epochs must be specified in the config"
		)

	if config.training.warmup_steps is None:
		config.training.warmup_steps = int(config.training.max_steps * 0.01)

	# Add ModelCheckpoint callback to save the checkpoint when validation loss is at a new low
	checkpoint_callback = ModelCheckpoint(
		monitor="train/total_loss",
		mode="min",
		save_top_k=config.training.save_top_k,
		save_last=True,
		filename="epoch-{epoch:02d}-train_loss-{train/total_loss:.4f}",
		dirpath=config.training.checkpoint_dir,
		# Don't use val_loss in filename for periodic saves - causes failures when val doesn't run
		auto_insert_metric_name=False
	)
	
	# Add separate callback for periodic saves (no val_loss dependency). Use
	# step-based saves for streaming datasets (save_every_n_steps) and epoch-based
	# saves otherwise (save_every_n_epochs); whichever the config provides.
	save_every_n_steps = config.training.get('save_every_n_steps', None)
	save_every_n_epochs = config.training.get('save_every_n_epochs', None)
	if save_every_n_steps is not None:
		periodic_checkpoint_callback = ModelCheckpoint(
			save_top_k=-1,  # Save all periodic checkpoints
			filename="step-{step:08d}",
			dirpath=config.training.checkpoint_dir,
			every_n_train_steps=save_every_n_steps,
			auto_insert_metric_name=False
		)
	elif save_every_n_epochs is not None:
		periodic_checkpoint_callback = ModelCheckpoint(
			save_top_k=-1,  # Save all periodic checkpoints
			filename="epoch-{epoch:02d}",
			dirpath=config.training.checkpoint_dir,
			every_n_epochs=save_every_n_epochs,
			auto_insert_metric_name=False
		)
	else:
		raise ValueError(
			"Either save_every_n_steps or save_every_n_epochs must be specified in the config"
		)

	trainer_kwargs["callbacks"] = [checkpoint_callback, periodic_checkpoint_callback]

	if wandb_logger is not None:
		trainer_kwargs["logger"] = wandb_logger

	trainer = pl.Trainer(**trainer_kwargs)

	# Train the model
	ckpt_path = None
	if "resume_path" in config.training:
		ckpt_path = config.training.resume_path
 	
	trainer.fit(module, 
             datamodule=data_module, 
             ckpt_path=ckpt_path)
 
	# Only finish wandb on rank 0
	if int(os.environ.get("LOCAL_RANK", 0)) == 0:
		wandb.finish()


if __name__ == '__main__':
	# Parse arguments to get config name
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_name', type=str, default='config',
	                   help='Name of the config file to use')
	parser.add_argument('--task', type=str, default=None,
	                   help='Task name (uses config_{task}.yaml)')
 
	# Parse known args (hydra will handle the rest)
	args, unknown = parser.parse_known_args()
	
	# Determine config name from task or config_name
	if args.task:
		config_name = f'config_{args.task}'
	else:
		config_name = args.config_name
	
	print(f"Using config: {config_name}.yaml")
	
	# Add config name to Hydra overrides (this persists across DDP subprocesses)
	if '--config-name' not in unknown and f'--config-name={config_name}' not in unknown:
		unknown.insert(0, f'--config-name={config_name}')
	
	# Reconstruct sys.argv for hydra
	sys.argv = [sys.argv[0]] + unknown
	
	# Define main function with default config (will be overridden by command line)
	@hydra.main(version_base=None,
	           config_path=CONFIG_DIR,
	           config_name='config')
	def main(config):
		"""Main entry point for training"""
		train(config)
	
	main()