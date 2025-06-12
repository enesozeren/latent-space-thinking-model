import argparse
import os
import logging

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.train.utils import (
    load_config, setup_logging
)
from src.train.lightning_modules import (
    SFTDataModule, GenerateSamplesCallback, ModelLightningModule, DatasetRefreshCallback
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train_model(config_path: str) -> None:
    """Main training routine using PyTorch Lightning."""
    cfg = load_config(config_path)
    
    # Setup logging and output directory
    output_dir = setup_logging(cfg)
    
    # Log the configuration file
    with open(config_path, "r") as f:
        config_content = f.read()
    logging.info("Configuration file contents:\n%s", config_content)
    
    # Create model
    model = ModelLightningModule(cfg)
    tokenizer = model.tokenizer  # Access tokenizer from the model
    
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
        every_n_train_steps=None,
        save_on_train_epoch_end=True # Only save at the end of each epoch
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create data module
    data_module = SFTDataModule(cfg, tokenizer)
    
    # Qualitative evaluation callback
    sample_cb = GenerateSamplesCallback(
        tokenizer=tokenizer,
        num_samples=2,
        is_latent_reasoner=cfg["model"]["is_latent_reasoner"]
    )
    callbacks.append(sample_cb)
    
    # Dataset refresh callback for epoch-based preprocessing changes
    dataset_refresh_cb = DatasetRefreshCallback()
    callbacks.append(dataset_refresh_cb)
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=cfg["training"]["num_train_epochs"],
        accelerator="auto",
        devices="auto",
        strategy="auto",  # Will automatically choose the best strategy for multi-GPU
        precision="bf16-mixed",  # Use bf16 precision
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
    parser = argparse.ArgumentParser(description="SFT using PyTorch Lightning")
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
