import yaml
import argparse
import os
import logging
import wandb
from datetime import datetime
import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer

from src.train.rewards import format_reward, latent_format_reward, accuracy_reward
from src.data_process.process_data import prepare_dataset
from src.latent_reasoner.model import LatentReasoner

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
        # Other args
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        log_completions=True,
        use_vllm=False
    )

def train_model(config_path: str) -> None:
    """Main training routine."""
    cfg = load_config(config_path)
    
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
    data = prepare_dataset(config=cfg, is_latent_reasoner=True)

    # Arguments
    args = setup_training_args(config=cfg)
    
    # Model and tokenizer
    # Then create the model with this config
    model = LatentReasoner.from_pretrained(cfg["model"]["base_model_name_or_path"])
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"])

    # Define new tokens for latent steps
    START = "<|start-latent|>"
    LAT   = "<|latent|>"
    END   = "<|end-latent|>"
    new_specials = [START, LAT, END]

    # Only add / resize / init if it is not the first training run
    if not all(tok in tokenizer.get_vocab() for tok in new_specials):
        # 3) Add them to the tokenizer’s vocab
        tokenizer.add_tokens(new_specials)
        model.resize_token_embeddings(len(tokenizer))  # expand model embeddings

        # 4) “Save” them as attributes for easy access
        tokenizer.start_latent_token   = START
        tokenizer.latent_token         = LAT
        tokenizer.end_latent_token     = END

        tokenizer.start_latent_token_id = tokenizer.convert_tokens_to_ids(START)
        tokenizer.latent_token_id       = tokenizer.convert_tokens_to_ids(LAT)
        tokenizer.end_latent_token_id   = tokenizer.convert_tokens_to_ids(END)

        # mirror them on your model
        model.start_latent_token_id = tokenizer.start_latent_token_id
        model.latent_token_id       = tokenizer.latent_token_id
        model.end_latent_token_id   = tokenizer.end_latent_token_id
        
        # Get the embedding layer correctly
        embedding_layer = model.get_input_embeddings()
        
        # Init the new latent tokens
        vocab = tokenizer.get_vocab()
        # Use torch.no_grad() to safely modify the weights
        with torch.no_grad():
            # copy existing tokens
            embedding_layer.weight[model.start_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
            embedding_layer.weight[model.end_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
    else:
        # tokens already there – still handy to have the ids on the objects
        tokenizer.start_latent_token_id = tokenizer.convert_tokens_to_ids(START)
        tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids(LAT)
        tokenizer.end_latent_token_id = tokenizer.convert_tokens_to_ids(END)
        model.start_latent_token_id = tokenizer.start_latent_token_id
        model.latent_token_id = tokenizer.latent_token_id
        model.end_latent_token_id = tokenizer.end_latent_token_id
        logging.info("Special tokens already present – skipping re-initialisation.")        
    
    # GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[latent_format_reward, accuracy_reward],
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        num_latent_steps=cfg["model"]["num_latent_steps"]
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using GRPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/latent_reasoner_rl.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args.config)
