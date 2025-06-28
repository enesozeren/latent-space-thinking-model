import argparse
import os
import logging

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.train.utils import (
    load_config, setup_logging, is_rank_zero
)
from src.train.lightning_modules import (
    SFTDataModule, GenerateSamplesCallback, ModelLightningModule, 
    DatasetRefreshCallback, HFModelCheckpoint
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
    logging.info("Model created.")

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
    checkpoint_callback = HFModelCheckpoint(
        output_dir, 
        is_latent_reasoner=cfg["model"]["is_latent_reasoner"]
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Create data module
    data_module = SFTDataModule(cfg, tokenizer)
    logging.info("Data created.")
    
    # If using latent reasoning, add dataset refresh callback to introduce latent steps incrementally
    if cfg["model"]["is_latent_reasoner"]:
        logging.info("Latent Reasoning enabled, adding dataset refresh callback.")
        # Dataset refresh callback
        dataset_refresh_cb = DatasetRefreshCallback(
            add_num_latents_per_update = cfg["training"]["add_num_latents_per_update"],
            num_tokens_per_latent = cfg["training"]["num_tokens_per_latent"]
        )
        callbacks.append(dataset_refresh_cb)

    # Qualitative evaluation callback
    sample_cb = GenerateSamplesCallback(
        num_samples=2,
        is_latent_reasoner=cfg["model"]["is_latent_reasoner"],
        add_num_latents_per_update=cfg["training"]["add_num_latents_per_update"] if cfg["model"]["is_latent_reasoner"] else None,
    )
    callbacks.append(sample_cb)        
    
    # Setup trainer
    trainer = L.Trainer(
        max_epochs=cfg["training"]["num_train_epochs"],
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-true",
        accumulate_grad_batches=cfg["training"]["gradient_accumulation_steps"],
        log_every_n_steps=cfg["training"]["logging_steps"],
        val_check_interval=cfg["training"]["eval_steps"],
        logger=logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        enable_checkpointing=is_rank_zero(),     # no duplicate ckpt dirs
        enable_progress_bar=is_rank_zero(),      # cleaner stdout
        enable_model_summary=is_rank_zero(),     # summary once
    )
    
    # Start training
    logging.info("SFT is starting.")
    trainer.fit(model, datamodule=data_module)
    
    # Save final model
    if is_rank_zero():
        final_model_path = os.path.join(output_dir, "final_model")
        model.model.save_pretrained(final_model_path)
        model.tokenizer.save_pretrained(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="SFT using PyTorch Lightning")
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/configs/latent_reasoner_sft.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args.config_path)
