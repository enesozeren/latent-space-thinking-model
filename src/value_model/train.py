# train_value_model.py
import os
import argparse
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.train.utils import load_config
from src.value_model.lightning_modules import H5ValueDataset, LigthningValueHeadModel

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
    pos_weight  = float(cfg.get("training", {}).get("pos_weight", 1.0))
    num_epochs  = int(cfg.get("training", {}).get("num_epochs", 10))
    val_ratio   = float(cfg.get("training", {}).get("val_ratio", 0.2))
    test_ratio  = float(cfg.get("training", {}).get("test_ratio", 0.1))  # New parameter
    seed = int(cfg.get("training", {}).get("seed", 42))
    data_path = cfg.get("dataset", {}).get("path")
    log_every_n_steps = cfg.get("logging", {}).get("log_every_n_steps", 100)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.get("logging", {}).get("output_dir"), timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # seed for reproducebility
    seed_everything(seed, workers=True)

    # data
    # dataset and splits  
    full_ds = H5ValueDataset(data_path)
    n_total = len(full_ds)
    
    # Calculate split sizes
    n_test = max(1, int(test_ratio * n_total))
    n_val = max(1, int(val_ratio * n_total))
    n_train = n_total - n_val - n_test
    
    # Ensure we have at least 1 sample in each split
    if n_train <= 0:
        raise ValueError(f"Training set would be empty. Total samples: {n_total}, val: {n_val}, test: {n_test}")
    
    # Create three-way split
    train_ds, val_ds, test_ds = random_split(
        full_ds,
        lengths=[n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_stats = get_stats(train_ds, "Train")
    val_stats = get_stats(val_ds, "Val")
    test_stats = get_stats(test_ds, "Test")

    # input dim
    input_dim = full_ds.latent_dim

    # model
    lit = LigthningValueHeadModel(
        input_dim=input_dim, hidden_dims=model_hidden_dims, 
        dropout=dropout, learning_rate=lr, pos_weight=pos_weight
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
            "dataset/test_size": test_stats["size"],
            "dataset/test_mean_reward": test_stats["mean_reward"],
            "dataset/total_size": len(full_ds),
            "dataset/val_ratio": val_ratio,
            "dataset/test_ratio": test_ratio
        })
        
        # Log model parameter statistics to wandb
        logger.experiment.log({
            "model/total_parameters": total_params,
            "model/input_dimension": input_dim
        })

    # loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )

    # Setup model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best_val_roc_auc_{epoch:02d}_{val_roc_auc:.4f}",
        monitor="val_roc_auc",
        mode="max",  # We want to maximize ROC AUC score
        save_top_k=1,  # Keep only the best checkpoint
        save_last=True,  # Also save the last checkpoint
        auto_insert_metric_name=False
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
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(lit, train_loader, val_loader)
    
    # Load the best checkpoint for final testing
    print(f"\nLoading best checkpoint: {checkpoint_callback.best_model_path}")
    best_model = LigthningValueHeadModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        input_dim=input_dim,
        hidden_dims=model_hidden_dims,
        dropout=dropout,
        learning_rate=lr,
        pos_weight=pos_weight
    )
    
    # Test the best model
    print("\n" + "="*50)
    print("EVALUATING BEST MODEL ON TEST SET")
    print(f"Best validation F1 score: {checkpoint_callback.best_model_score:.4f}")
    print("="*50)
    
    test_results = trainer.test(best_model, test_loader, verbose=True)
    
    # Save the raw PyTorch model (without Lightning wrapper)
    raw_model_path = os.path.join(output_dir, "best_model.ckpt")
    torch.save(best_model.model, raw_model_path)
    print(f"\nRaw PyTorch model saved to: {raw_model_path}")
    
    # Log test results to wandb if logger is available
    if logger is not None:
        test_metrics = test_results[0]  # trainer.test returns a list with one dict
        logger.experiment.log({
            f"final_test/{k}": v for k, v in test_metrics.items()
        })
        # Also log the best validation F1 score
        logger.experiment.log({
            "best_val_f1": checkpoint_callback.best_model_score
        })
    
    print(f"\nBest validation F1 score: {checkpoint_callback.best_model_score:.4f}")
    print(f"Best checkpoint saved at: {checkpoint_callback.best_model_path}")
    print(f"\nFinal test results:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()