import yaml
import argparse
import os
import logging
import wandb
from datetime import datetime
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.train_sft_grpo.rewards import latent_format_reward, accuracy_reward
from src.data_process.process_data import prepare_dataset_rl
from src.latent_reasoner.model import LatentReasoner
from src.train_sft_grpo.utils import load_config, setup_latent_tokens
from src.latent_rl.latent_rl_trainer import LatentRLTrainer
from src.value_model.model import ValueHeadModel, ValueModel


def train_model(config_path: str) -> None:
    """Main training routine."""
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # figure out which process we're in
    rank = int(os.environ.get("RANK", 0))
    is_main = (rank == 0)

    # only in the rank‑0 process
    if is_main and "wandb" in cfg and cfg["wandb"].get("project"):
        run = wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )
        # this will pin your YAML into the run's Files tab
        wandb.save(config_path)
    else:
        os.environ["WANDB_MODE"] = "disabled"
        
    # Logging
    base_output_dir = cfg["training"]["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    # Update cfg for output_dir
    cfg["training"]["output_dir"] = output_dir

    # Ensure the output directory exists before writing the log file.
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
    
    # Log the configuration file
    with open(config_path, "r") as f:
        config_content = f.read()
    logging.info("Configuration file contents:\n%s", config_content)
    
    # Prepare the dataset
    data = prepare_dataset_rl(config=cfg, is_latent_reasoner=True)
    # Create DataLoader directly from the train split
    train_loader = DataLoader(
        data["train"],
        batch_size=cfg["training"]["per_device_train_batch_size"],
        shuffle=False
    )    

    # Model and tokenizer
    # Then create the model with this config
    model = LatentReasoner.from_pretrained(cfg["model"]["base_model_name_or_path"]).to(device)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"])

    # Set up special tokens for think, answer and latent reasoning
    model, tokenizer = setup_latent_tokens(model=model, 
                                            tokenizer=tokenizer, 
                                            is_latent_reasoner=cfg["model"]["is_latent_reasoner"])
    
    # generation config
    generation_config = cfg["generation_parameters"]

    # Value model with pre-trained value head
    pretrained_value_head = torch.load(
        cfg["value_head_model"]["pretrained_value_head_model_path"], 
        map_location=device, weights_only=False
    )
    value_model = ValueModel(latent_reasoner=model, value_head=pretrained_value_head)

    # GRPO trainer
    trainer = LatentRLTrainer(
        model=model,
        value_model=value_model,
        tokenizer=tokenizer,
        reward_funcs=[latent_format_reward, accuracy_reward],
        args=cfg["training"],
        train_loader=train_loader,
        num_latent_steps=cfg["model"]["num_latent_steps"],
        generation_config=generation_config,
        device=device
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using Latent RL algorithm.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/latent_rl/latent_reasoner_sft_latent_rl.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args.config)
