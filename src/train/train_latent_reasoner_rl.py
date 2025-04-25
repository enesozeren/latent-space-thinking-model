import yaml
import argparse
import os
import logging
import wandb
from datetime import datetime
from trl import GRPOTrainer, GRPOConfig

# Replace standard imports with our LatentReasoner
from src.latent_reasoner.model import LatentReasoner

from src.train.rewards import format_reward, accuracy_reward
from src.data_process.process_data import prepare_dataset

def load_config(config_path):
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_training_args(config: dict) -> GRPOConfig:
    """Translate YAML `training` + `grpo` sections into a GRPOConfig."""
    return GRPOConfig(
        seed=config["training"]["seed"],
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        lr_scheduler_type=config["training"]["lr_scheduler_type"],
        lr_scheduler_kwargs=config["training"]["lr_scheduler_kwargs"],
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        # GRPO specific hps
        beta=config["grpo"]["beta"],
        epsilon=config["grpo"]["epsilon"],
        scale_rewards=config["grpo"]["scale_rewards"],
        num_generations=config["grpo"]["num_generations"],
        max_completion_length=config["grpo"]["max_completion_length"],
        temperature=config["grpo"]["temperature"],
        run_name=config.get("wandb", {}).get("run_name"),
        bf16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        log_completions=True,
        use_vllm=False # Since we use a custom model
    )

def train_model(config_path: str) -> None:
    """Main training routine."""
    cfg = load_config(config_path)
    rank = int(os.environ.get("RANK", 0))
    is_main = (rank == 0)

    # Initialize W&B on main process
    if is_main and cfg.get("wandb", {}).get("project"):
        run = wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )
        wandb.save(config_path)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # Setup logging
    base_output_dir = cfg["training"]["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    cfg["training"]["output_dir"] = output_dir
    log_path = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info("Logging initialized – saving to %s", log_path)
    with open(config_path, "r") as f:
        logging.info("Configuration file contents:\n%s", f.read())

    # Prepare data
    data = prepare_dataset(cfg)

    # GRPO args
    args = setup_training_args(cfg)

    # Instantiate our LatentReasoner
    latent = LatentReasoner(model_name=cfg["model"]["model_name_or_path"])
    model = latent.model
    tokenizer = latent.tokenizer
    # Monkey-patch generate to include latent loop
    model.generate = latent.generate

    # Setup trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=args,
        train_dataset=data["train"],
        eval_dataset=data.get("validation")
    )

    # Train
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using GRPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/latent_reasoner_qwen2_1p5b_it_rl.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    train_model(args.config)
