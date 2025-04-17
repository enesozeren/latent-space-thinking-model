import yaml
import argparse
import re
import os
import logging
from datetime import datetime
from datasets import load_dataset, DatasetDict
from trl import GRPOTrainer, GRPOConfig
from src.train.rewards import format_reward, accuracy_reward
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts.prompts import SYSTEM_PROMPT

# --------------------------------------------------------------
#  NOTE
#  -----
#  • Launch with Hugging Face Accelerate, e.g.:
#      accelerate launch --num_processes 4 src/train/train_rl.py
# --------------------------------------------------------------

def load_config(config_path):
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def _extract_gsm8k_answer(raw_answer: str) -> str:
    """
    Extract the canonical short answer from a GSM8K solution string.

    GSM8K places the final numeric answer after the delimiter '####', e.g.
        "... reasoning ...\n#### 24"
    We take everything after the *last* occurrence of that delimiter,
    strip whitespace, and drop a trailing period if present.
    """
    if "####" in raw_answer:
        answer = raw_answer.split("####")[-1]
    else:
        answer = raw_answer
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    return answer

def _gsm8k_to_grpo(example: dict) -> dict:
    """Convert a GSM8K row → GRPO expected format using SYSTEM_PROMPT."""
    question = example["question"].strip()
    answer = _extract_gsm8k_answer(example["answer"])
    return {
        "prompt": SYSTEM_PROMPT.format(user_input=question),
        "answer": answer,
    }

def prepare_dataset(config: dict) -> DatasetDict:
    """Load GSM8K and re‑format into the columns GRPOTrainer expects."""
    raw_ds = load_dataset(
        config["dataset"]["name"],
        config["dataset"].get("subname", None),
    )

    # Train/val split
    split_ds = raw_ds["train"].train_test_split(test_size=0.1, seed=42)

    processed = {
        "train": split_ds["train"].map(_gsm8k_to_grpo, remove_columns=raw_ds["train"].column_names),
        "validation": split_ds["test"].map(_gsm8k_to_grpo, remove_columns=raw_ds["train"].column_names),
        "test": raw_ds["test"].map(_gsm8k_to_grpo, remove_columns=raw_ds["test"].column_names),
    }

    return DatasetDict(processed)

def setup_training_args(config: dict) -> GRPOConfig:
    """Translate YAML `training` + `grpo` sections into a GRPOConfig."""
    return GRPOConfig(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
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
        ddp_find_unused_parameters=False,
        log_completions=True
    )

def train_model(config_path: str) -> None:
    """Main training routine."""
    cfg = load_config(config_path)
    
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
    
    # Weights & Biases: only set the project so HF/TRL auto‑initialises
    if "wandb" in cfg and cfg["wandb"].get("project"):
        os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    data = prepare_dataset(cfg)
    args = setup_training_args(cfg)

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["model_name_or_path"], attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["model_name_or_path"])

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
    )

    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using GRPO.")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/gemma_3_1b_it_rl.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()

if __name__ == "__main__":
    cli_args = parse_args()
    train_model(cli_args.config)
