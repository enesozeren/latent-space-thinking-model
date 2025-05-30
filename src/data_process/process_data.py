import re
import torch
from datasets import load_dataset, DatasetDict
from prompts.prompts import (
    SYSTEM_PROMPT, 
    SYSTEM_PROMPT_LATENT_REASONER
)
import logging

def prepare_dataset(config: dict, is_latent_reasoner: bool) -> DatasetDict:
    """Load OpenR1-Math-220k dataset and re-format into the columns GRPOTrainer expects."""
    raw_ds = load_dataset(config["dataset"]["name"], "default")
    
    # Since there's only a train split in DeepMath-103K, create train/val splits
    split_ds = raw_ds["train"].train_test_split(test_size=0.1, seed=config["training"]["seed"])
    
    processed = {
        "train": split_ds["train"].map(
            lambda x: _openr1_to_grpo(x, is_latent_reasoner), 
            remove_columns=raw_ds["train"].column_names
        ),
        "validation": split_ds["test"].map(
            lambda x: _openr1_to_grpo(x, is_latent_reasoner), 
            remove_columns=raw_ds["train"].column_names
        )
    }
    
    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info(f"Is LatR?: {is_latent_reasoner}")    
    logger.info("Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")

    return DatasetDict(processed)

def _openr1_to_grpo(example: dict, is_latent_reasoner: bool) -> dict:
    """Convert an OpenR1-Math-220k row → GRPO expected format.
    
    Args:
        example: An OpenR1-Math-220k example
    """
    question = example["problem"].strip()
    answer = example["answer"].strip()
    sys_prompt = SYSTEM_PROMPT if not is_latent_reasoner else SYSTEM_PROMPT_LATENT_REASONER
    prompt = sys_prompt + "\nUser:" + question + "\nAssistant:"

    return {
        "prompt": prompt,
        "answer": answer,
    }

def _gsm8k_to_latent_reasoning_sft(example, tokenizer, max_num_latent_steps):
    """Process a single GSM8K example into SFT format with latent-reasoning tokens,
    returning input_ids, attention_mask, labels, and an optional label_mask."""
    question = example["question"].strip()
    # Convert trailing '#### answer' to '\\boxed{answer}'
    raw_answer = example.get("answer", "").strip()
    # Replace '\n#### <value>' or '#### <value>' at end of string
    answer = re.sub(r"(?:\\n)?####\s*(.+)$", r"\\boxed{\1}", raw_answer)

    # Randomly determine the number of latent steps (between 1 and max_num_latent_steps)
    num_latent_steps = torch.randint(1, max_num_latent_steps + 1, (1,)).item()

    # Build the latent token sequence
    latent_ids = [tokenizer.start_latent_token_id] + \
                 [tokenizer.latent_token_id] * num_latent_steps + \
                 [tokenizer.end_latent_token_id]

    # Tokenize prefix and answer separately
    prefix_text = "\nUser:" + question + "\nAssistant:"
    prefix_tokens = tokenizer(prefix_text, add_special_tokens=False).input_ids
    answer_tokens = tokenizer(answer, add_special_tokens=False).input_ids

    # Combine all token ids: prefix + latent tokens + answer
    input_ids = prefix_tokens + latent_ids + answer_tokens
    attention_mask = [1] * len(input_ids)

    # Create labels: mask prefix and latent with -100 so loss is only computed on answer tokens
    labels = [-100] * (len(prefix_tokens) + len(latent_ids)) + answer_tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def prepare_dataset_latent_reasoning_sft(config: dict, tokenizer) -> DatasetDict:
    """Convert GSM8K into latent-reasoning SFT format using the provided tokenizer."""
    # Load dataset
    raw_ds = load_dataset(config["dataset"]["name"], "main", split="train")

    # Create train/validation splits
    split_ds = raw_ds.train_test_split(test_size=0.1, seed=config["training"]["seed"])

    # Get the number of latent steps from config (default 3)
    max_num_latent_steps = config["training"]["max_num_latent_steps"]

    # Process both splits with map, passing tokenizer & num_latent_steps via fn_kwargs
    processed = {
        "train": split_ds["train"].map(
            _gsm8k_to_latent_reasoning_sft,
            fn_kwargs={"tokenizer": tokenizer, "max_num_latent_steps": max_num_latent_steps},
            remove_columns=raw_ds.column_names,
        ),
        "validation": split_ds["test"].map(
            _gsm8k_to_latent_reasoning_sft,
            fn_kwargs={"tokenizer": tokenizer, "max_num_latent_steps": max_num_latent_steps},
            remove_columns=raw_ds.column_names,
        ),
    }

    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info("GSM8K SFT Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")
    logger.info(f"  Max number of latent steps: {max_num_latent_steps}")

    return DatasetDict(processed)
