import yaml
import argparse
import torch
from datasets import load_dataset, DatasetDict
from trl import GRPOTrainer, GRPOConfig
from src.train.rewards import (
    accuracy_reward_func,
    format_reward_func,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path):
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(config):
    """Prepare and split the dataset according to configuration."""
    # Load the dataset
    gsm = load_dataset(
        config["dataset"]["name"],
        config["dataset"]["subname"]
    )

    # Split the train set into train and validation
    split_dataset = gsm['train'].train_test_split(test_size=0.1, seed=42)

    # Combine the new splits with the original test set
    return DatasetDict({
        'train': split_dataset['train'],
        'validation': split_dataset['test'],
        'test': gsm['test']
    })


def setup_training_args(config):
    """Set up and return the GRPO configuration."""
    training_args = GRPOConfig(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        max_steps=config["training"]["max_steps"],
        logging_steps=config["training"]["logging_steps"],
        eval_steps=config["training"]["eval_steps"],
        save_steps=config["training"]["save_steps"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        
        # GRPO specific parameters
        beta=config["grpo"]["beta"],
        epsilon=config["grpo"]["epsilon"],
        scale_rewards=config["grpo"]["scale_rewards"],

        # Generation parameters
        num_generations =config["grpo"]["num_generations"],
        max_completion_length=config["grpo"]["max_completion_length"],
        temperature=config["grpo"]["temperature"],
        
        # GPU training settings
        fp16=torch.cuda.is_available(),  # Use mixed precision training if CUDA available
        dataloader_num_workers=4,
    )
    
    return training_args


def train_model(config_path):
    """Main training function that orchestrates the entire process."""
    # Load configuration
    config = load_config(config_path)
    
    # Prepare dataset
    gsm = prepare_dataset(config)
    
    # Set up training arguments
    training_args = setup_training_args(config)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(config["model"]["model_name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["model_name_or_path"])
    
    # Apply data parallelism if multiple GPUs are available
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Using DataParallel with {gpu_count} GPUs")
            model = torch.nn.DataParallel(model)
            model = model.to('cuda')
    
    # Initialize trainer with our parallelized model
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[accuracy_reward_func, format_reward_func],
        args=training_args,
        train_dataset=gsm['train'],
        eval_dataset=gsm['validation']
    )
    
    # Start training
    trainer.train()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model using GRPO.')
    parser.add_argument('--config', type=str, default="src/configs/gemma_3_1b_pt_rl.yaml",
                        help='Path to the configuration YAML file')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.config)