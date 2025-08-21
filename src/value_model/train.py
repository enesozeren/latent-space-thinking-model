# train_value_model.py
import os
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger

from src.train.utils import load_config
from src.value_model.lightning_modules import H5ValueDataset, LigthningValueModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train ValueModel on HDF5 latent vectors and accuracy rewards.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/value_model/value_model_training.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


def get_stats(ds, name):
    # ds is a Subset, so access the underlying dataset and indices
    x_data = ds.dataset.y_data[ds.indices]
    mean_reward = x_data.mean().item()
    size = len(ds)
    print(f"{name} set size: {size}")
    print(f"{name} set mean accuracy reward: {mean_reward:.4f}")
    return {"size": size, "mean_reward": mean_reward}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # hyperparameters
    model_hidden_dims = cfg.get("model", {}).get("hidden_dims", 256)
    dropout  = float(cfg.get("training", {}).get("dropout", 0.1))
    batch_size  = int(cfg.get("training", {}).get("batch_size", 256))
    lr          = float(cfg.get("training", {}).get("learning_rate", 1e-2))
    num_epochs  = int(cfg.get("training", {}).get("num_epochs", 10))
    val_ratio   = float(cfg.get("training", {}).get("val_ratio", 0.2))
    seed = int(cfg.get("training", {}).get("seed", 42))
    data_path = cfg.get("dataset", {}).get("path")
    log_every_n_steps = cfg.get("logging", {}).get("log_every_n_steps", 100)
    
    output_dir = cfg.get("logging", {}).get("output_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    # seed for reproducebility
    seed_everything(seed, workers=True)

    # data
    # dataset and splits  
    full_ds = H5ValueDataset(data_path)
    n_total = len(full_ds)
    n_val = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_stats = get_stats(train_ds, "Train")
    val_stats = get_stats(val_ds, "Val")

    # input dim
    input_dim = full_ds.latent_dim

    # model
    lit = LigthningValueModel(
        input_dim=input_dim, hidden_dims=model_hidden_dims, 
        dropout=dropout, learning_rate=lr
    )

    # Calculate model parameter count
    total_params = sum(p.numel() for p in lit.model.parameters())

    # Setup wandb logger if configured
    logger = None
    if "wandb" in cfg and cfg["wandb"].get("project"):
        logger = WandbLogger(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("run_name"),
            save_dir=output_dir,
            config=cfg
        )
        
        # Log dataset statistics to wandb
        logger.experiment.log({
            "dataset/train_size": train_stats["size"],
            "dataset/train_mean_reward": train_stats["mean_reward"],
            "dataset/val_size": val_stats["size"],
            "dataset/val_mean_reward": val_stats["mean_reward"],
            "dataset/total_size": len(full_ds),
            "dataset/val_ratio": val_ratio
        })
        
        # Log model parameter statistics to wandb
        logger.experiment.log({
            "model/total_parameters": total_params,
            "model/input_dimension": input_dim
        })

    # loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True, drop_last=False
    )

    # trainer
    accelerator = "gpu" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=num_epochs,
        log_every_n_steps=log_every_n_steps,
        deterministic=True,
        default_root_dir=output_dir,
        logger=logger,
    )

    trainer.fit(lit, train_loader, val_loader)


if __name__ == "__main__":
    main()
