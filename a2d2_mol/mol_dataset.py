#!/usr/bin/env python
"""
Adapter to use HuggingFace datasets with the any-length discrete diffusion model.
This module converts HuggingFace datasets (like datamol-io/safe-drugs) into the format
expected by the training pipeline.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
from safe.tokenizer import SAFETokenizer
from mol_utils.bracket_safe_converter import safe2bracketsafe
from typing import Optional, List
import re


def get_tokenizer():
    """Get SAFE tokenizer with added special tokens."""
    tk = SAFETokenizer.from_pretrained('datamol-io/safe-gpt').get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk


class Collator:
    """Data collator for SAFE/bracket-SAFE format."""
    
    def __init__(self, config, tokenizer=None):
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.max_length = config.interpolant.max_length
        self.use_bracket_safe = config.training.get('use_bracket_safe', False)
    
    def __call__(self, examples):
        # Handle both dict with 'labels' and direct string format
        inputs = []
        for example in examples:
            if isinstance(example, dict):
                # Try different key names: 'input', 'labels', 'smiles'
                input_text = example.get('input', example.get('labels', example.get('smiles', '')))
            else:
                input_text = example
            
            if self.use_bracket_safe:
                input_text = safe2bracketsafe(input_text)
            
            inputs.append(input_text)
        
        batch = self.tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Convert BatchEncoding to plain dict with tensors
        # Remove token_type_ids if present (not needed for diffusion models)
        result = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        return result


class HFDatasetAdapter(Dataset):
    """Adapts HuggingFace datasets to the format expected by the diffusion model."""
    
    def __init__(self, hf_dataset, tokenizer, smiles_column='smiles', max_length=1024, convert_to_safe=False, is_streaming=False):
        """
        Args:
            hf_dataset: HuggingFace dataset object (streaming or regular)
            tokenizer: SMILES tokenizer instance
            smiles_column: Name of the column containing SMILES strings
            max_length: Maximum sequence length
            convert_to_safe: Whether to convert SMILES to SAFE format
            is_streaming: Whether dataset is in streaming mode
        """
        self.tokenizer = tokenizer
        self.smiles_column = smiles_column
        self.max_length = max_length
        self.convert_to_safe = convert_to_safe
        self.is_streaming = is_streaming
        
        if is_streaming:
            # For streaming datasets, we don't pre-load the data
            self.data = hf_dataset
            self._length = None  # Unknown length for streaming
            print(f'Initialized streaming dataset adapter')
        else:
            # Store raw data without pre-tokenization (tokenization will happen in collator)
            print(f'Initializing HF dataset adapter with {len(hf_dataset)} samples...')
            self.data = []
            for item in hf_dataset:
                smiles = item[smiles_column]
                if smiles:  # Skip empty SMILES
                    self.data.append({'input': smiles, 'labels': smiles})
            print(f'Processed {len(self.data)} valid samples')
    
    def __len__(self):
        if self.is_streaming:
            # Streaming datasets don't have a length
            # Return a large number to prevent issues with samplers
            return 10_000_000 if self._length is None else self._length
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_streaming:
            # For streaming, iteration happens differently
            raise NotImplementedError("Streaming datasets should be iterated, not indexed")
        return self.data[idx]
    
    def __iter__(self):
        """Support iteration for streaming datasets."""
        if self.is_streaming:
            for item in self.data:
                smiles = item[self.smiles_column]
                if smiles:  # Skip empty SMILES
                    yield {'input': smiles, 'labels': smiles}
        else:
            for item in self.data:
                yield item


class HFDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for HuggingFace datasets."""
    
    def __init__(
        self,
        config,
        dataset_name: str,
        tokenizer: SAFETokenizer,
        smiles_column: str = 'smiles',
        val_split: float = 0.1,
        test_split: Optional[float] = None,
        streaming: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ):
        """
        Args:
            config: Configuration object containing training parameters
            dataset_name: HuggingFace dataset identifier (e.g., "datamol-io/safe-gpt")
            tokenizer: SMILES tokenizer instance
            smiles_column: Name of column containing SMILES strings
            val_split: Fraction of data to use for validation
            test_split: Optional fraction of data to use for testing
            streaming: Whether to use streaming mode (recommended for large datasets)
            max_train_samples: Maximum number of training samples to use (for non-streaming)
            max_val_samples: Maximum number of validation samples to use (for non-streaming)
        """
        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.smiles_column = smiles_column
        self.max_length = config.interpolant.max_length
        self.batch_size = config.training.per_gpu_batch_size
        self.num_workers = config.training.get('cpus', 4)
        self.val_split = val_split
        self.test_split = test_split
        self.streaming = streaming
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize collator
        self.collator = Collator(config, tokenizer)
    
    def setup(self, stage: Optional[str] = None):
        """Load and split the dataset."""
        print(f'Loading dataset: {self.dataset_name} (streaming={self.streaming})')
        
        if self.streaming:
            # Load dataset in streaming mode
            raw_dataset = load_dataset(self.dataset_name, streaming=True)
            
            # Handle different dataset structures
            if 'train' in raw_dataset:
                train_stream = raw_dataset['train']
            else:
                # If no splits exist, use the entire dataset
                train_stream = raw_dataset[list(raw_dataset.keys())[0]]
            
            # For streaming, we need to manually split train/val
            # Skip validation samples, then take training samples
            val_size = int(100000 * self.val_split)  # Assume ~100k samples for val split calculation
            train_size = 100000 - val_size
            
            # Create validation stream (take first val_size samples)
            val_stream = train_stream.take(val_size)
            
            # Create training stream (skip val_size samples, then iterate)
            train_stream_shifted = train_stream.skip(val_size)
            
            # Create adapted datasets
            self.train_dataset = HFDatasetAdapter(
                train_stream_shifted,
                self.tokenizer,
                self.smiles_column,
                self.max_length,
                is_streaming=True
            )
            
            self.val_dataset = HFDatasetAdapter(
                val_stream,
                self.tokenizer,
                self.smiles_column,
                self.max_length,
                is_streaming=True
            )
            
            print(f'Streaming dataset initialized - samples will be loaded on-the-fly')
            
        else:
            # Traditional non-streaming mode with full dataset loading
            raw_dataset = load_dataset(self.dataset_name)
            
            # Handle different dataset structures
            if 'train' in raw_dataset:
                train_data = raw_dataset['train']
            else:
                # If no splits exist, use the entire dataset and split it
                train_data = raw_dataset[list(raw_dataset.keys())[0]]
            
            # Limit samples if specified
            if self.max_train_samples:
                train_data = train_data.select(range(min(self.max_train_samples, len(train_data))))
            
            # Check if dataset already has validation split
            if 'validation' in raw_dataset or 'val' in raw_dataset:
                val_key = 'validation' if 'validation' in raw_dataset else 'val'
                val_data = raw_dataset[val_key]
            else:
                # Create train/val split
                split_dataset = train_data.train_test_split(test_size=self.val_split, seed=42)
                train_data = split_dataset['train']
                val_data = split_dataset['test']
            
            # Limit validation samples if specified
            if self.max_val_samples:
                val_data = val_data.select(range(min(self.max_val_samples, len(val_data))))
            
            # Create test split if requested
            if self.test_split and 'test' not in raw_dataset:
                split_dataset = train_data.train_test_split(test_size=self.test_split, seed=42)
                train_data = split_dataset['train']
                self.test_dataset = HFDatasetAdapter(
                    split_dataset['test'],
                    self.tokenizer,
                    self.smiles_column,
                    self.max_length,
                    is_streaming=False
                )
            elif 'test' in raw_dataset:
                self.test_dataset = HFDatasetAdapter(
                    raw_dataset['test'],
                    self.tokenizer,
                    self.smiles_column,
                    self.max_length,
                    is_streaming=False
                )
            
            # Create adapted datasets
            self.train_dataset = HFDatasetAdapter(
                train_data,
                self.tokenizer,
                self.smiles_column,
                self.max_length,
                is_streaming=False
            )
            
            self.val_dataset = HFDatasetAdapter(
                val_data,
                self.tokenizer,
                self.smiles_column,
                self.max_length,
                is_streaming=False
            )
            
            print(f'Dataset splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}')
            if self.test_dataset:
                print(f'Test: {len(self.test_dataset)}')
    
    def train_dataloader(self):
        if self.streaming:
            # Pass streaming dataset directly to DataLoader (HF IterableDataset)
            # Must use num_workers=0 when using .skip() or .take() operations
            return DataLoader(
                self.train_dataset.data,  # Use the raw HF streaming dataset
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=0,  # Required for streaming with skip/take operations
                pin_memory=True,
                shuffle=False,  # Cannot shuffle streaming datasets
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
    
    def val_dataloader(self):
        if self.streaming:
            # Pass streaming dataset directly to DataLoader (HF IterableDataset)
            # Must use num_workers=0 when using .skip() or .take() operations
            return DataLoader(
                self.val_dataset.data,  # Use the raw HF streaming dataset
                batch_size=self.batch_size,
                collate_fn=self.collator,
                num_workers=0,  # Required for streaming with skip/take operations
                pin_memory=True,
                shuffle=False,  # Cannot shuffle streaming datasets
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
    
    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collator,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
        return None


def setup_hf_data_and_update_config(config, dataset_name="datamol-io/safe-gpt", smiles_column="smiles", streaming=True):
    """
    Setup HuggingFace dataset and update config with token information.
    
    Args:
        config: Hydra config object
        dataset_name: HuggingFace dataset identifier
        smiles_column: Name of column containing SMILES strings
        streaming: Whether to use streaming mode (recommended for large datasets like safe-gpt)
    
    Returns:
        HFDataModule instance
    """
    # Initialize tokenizer
    tokenizer = get_tokenizer()
    
    # Update config with tokenizer info
    config.interpolant.tokens = len(tokenizer)
    config.interpolant.pad_token = tokenizer.pad_token_id
    config.interpolant.mask_token = tokenizer.mask_token_id
    
    # Create data module
    data_module = HFDataModule(
        config=config,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        smiles_column=smiles_column,
        val_split=0.1,
        streaming=streaming,
    )
    
    return data_module
