import yaml
import argparse
import os
import logging
import wandb
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data_process.process_data import prepare_dataset_latent_reasoning_sft
from src.latent_reasoner.model import LatentReasoner


class LatentReasonerDataset(Dataset):
    """Dataset wrapper for Lightning DataLoader."""
    
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function to pad sequences to the same length."""
    # Get max length in the batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad sequences
    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for item in batch:
        input_len = len(item['input_ids'])
        pad_len = max_len - input_len
        
        # Pad input_ids and attention_mask with pad_token_id (0)
        padded_batch['input_ids'].append(
            F.pad(item['input_ids'], (0, pad_len), value=0)
        )
        padded_batch['attention_mask'].append(
            F.pad(item['attention_mask'], (0, pad_len), value=0)
        )
        # Pad labels with -100 (ignore_index)
        padded_batch['labels'].append(
            F.pad(item['labels'], (0, pad_len), value=-100)
        )
    
    # Stack tensors
    return {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels'])
    }


class LatentReasonerLightningModule(L.LightningModule):
    """PyTorch Lightning module for training LatentReasoner with SFT."""
    
    def __init__(self, config: Dict[str, Any], tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.save_hyperparameters(config)
        
        # Initialize model
        self.model = LatentReasoner.from_pretrained(config["model"]["base_model_name_or_path"])
        
        # Setup special tokens
        self._setup_special_tokens()
        
        # Store training config for optimizer setup
        self.learning_rate = float(config["training"]["learning_rate"])
        self.weight_decay = config["training"]["weight_decay"]
        self.lr_scheduler_step_size = config["training"]["lr_scheduler_step_size"]
        self.lr_scheduler_gamma = config["training"]["lr_scheduler_gamma"]
        
    def _setup_special_tokens(self):
        """Setup special tokens for latent reasoning."""
        START = "<|start-latent|>"
        LAT = "<|latent|>"
        END = "<|end-latent|>"
        new_specials = [START, LAT, END]

        # Only add / resize / init if it is not the first training run
        if not all(tok in self.tokenizer.get_vocab() for tok in new_specials):
            # Add them to the tokenizer's vocab
            self.tokenizer.add_tokens(new_specials)
            self.model.resize_token_embeddings(len(self.tokenizer))  # expand model embeddings

            # Save them as attributes for easy access
            self.tokenizer.start_latent_token = START
            self.tokenizer.latent_token = LAT
            self.tokenizer.end_latent_token = END

            self.tokenizer.start_latent_token_id = self.tokenizer.convert_tokens_to_ids(START)
            self.tokenizer.latent_token_id = self.tokenizer.convert_tokens_to_ids(LAT)
            self.tokenizer.end_latent_token_id = self.tokenizer.convert_tokens_to_ids(END)

            # mirror them on your model
            self.model.start_latent_token_id = self.tokenizer.start_latent_token_id
            self.model.latent_token_id = self.tokenizer.latent_token_id
            self.model.end_latent_token_id = self.tokenizer.end_latent_token_id
            
            # Get the embedding layer correctly
            embedding_layer = self.model.get_input_embeddings()
            
            # Init the new latent tokens
            vocab = self.tokenizer.get_vocab()
            # Use torch.no_grad() to safely modify the weights
            with torch.no_grad():
                # copy existing tokens
                embedding_layer.weight[self.model.start_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
                embedding_layer.weight[self.model.end_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
        else:
            # tokens already there – still handy to have the ids on the objects
            self.tokenizer.start_latent_token_id = self.tokenizer.convert_tokens_to_ids(START)
            self.tokenizer.latent_token_id = self.tokenizer.convert_tokens_to_ids(LAT)
            self.tokenizer.end_latent_token_id = self.tokenizer.convert_tokens_to_ids(END)
            self.model.start_latent_token_id = self.tokenizer.start_latent_token_id
            self.model.latent_token_id = self.tokenizer.latent_token_id
            self.model.end_latent_token_id = self.tokenizer.end_latent_token_id
            logging.info("Special tokens already present – skipping re-initialisation.")

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        return self.model.sft_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["training"]["lr_scheduler_step_size"],
            gamma=self.config["training"]["lr_scheduler_gamma"]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class LatentReasonerDataModule(L.LightningDataModule):
    """Lightning DataModule for LatentReasoner training."""
    
    def __init__(self, config: Dict[str, Any], tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.batch_size = config["training"]["per_device_train_batch_size"]
        self.eval_batch_size = config["training"]["per_device_eval_batch_size"]
        
    def setup(self, stage: str):
        """Setup datasets."""
        # Prepare the dataset
        data = prepare_dataset_latent_reasoning_sft(config=self.config, tokenizer=self.tokenizer)
        
        self.train_dataset = LatentReasonerDataset(data["train"])
        self.val_dataset = LatentReasonerDataset(data["validation"])

        # 3. 👀  Show one training example for sanity-check
        if len(self.train_dataset):                      # just in case
            sample = self.train_dataset[0]

            # decode the input text so it’s readable
            decoded_prompt = self.tokenizer.decode(
                sample["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            print("\n=== Latent-Reasoner sanity check ===")
            print("Prompt:\n", decoded_prompt)
            # if you store the answer/label under another key, adjust here
            if "labels" in sample:
                print("\nLabel IDs:", sample["labels"])
            print("====================================\n")        

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )


def load_config(config_path):
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Setup logging configuration."""
    base_output_dir = config["training"]["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    
    # Update config for output_dir
    config["training"]["output_dir"] = output_dir

    # Ensure the output directory exists before writing the log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ],
        force=True  # override any previous logging configuration
    )

    logging.info("Logging initialised – saving to %s", log_path)
    return output_dir


def train_model(config_path: str) -> None:
    """Main training routine using PyTorch Lightning."""
    cfg = load_config(config_path)
    
    # Setup logging and output directory
    output_dir = setup_logging(cfg)
    
    # Log the configuration file
    with open(config_path, "r") as f:
        config_content = f.read()
    logging.info("Configuration file contents:\n%s", config_content)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"])
    
    # Setup wandb logger if configured
    logger = None
    if "wandb" in cfg and cfg["wandb"].get("project"):
        logger = WandbLogger(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("run_name"),
            save_dir=output_dir,
            config=cfg
        )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="checkpoint-{epoch:02d}-{step}",
        save_top_k=1, # Save top model based on validation loss
        every_n_train_steps=cfg["training"]["save_steps"],
        save_on_train_epoch_end=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create data module
    data_module = LatentReasonerDataModule(cfg, tokenizer)
    
    # Create model
    model = LatentReasonerLightningModule(cfg, tokenizer)
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=cfg["training"]["num_train_epochs"],
        accelerator="auto",
        devices="auto",
        strategy="auto",  # Will automatically choose the best strategy for multi-GPU
        precision="bf16-mixed",  # Use bf16 precision
        # gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=cfg["training"]["gradient_accumulation_steps"],
        log_every_n_steps=cfg["training"]["logging_steps"],
        val_check_interval=cfg["training"]["eval_steps"],
        logger=logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Start training
    trainer.fit(model, datamodule=data_module)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model")
    model.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"Final model saved to {final_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Latent Reasoner with SFT using PyTorch Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/latent_reasoner_sft.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args.config)
