from datasets import load_dataset, DatasetDict
from prompts.prompts import (SYSTEM_PROMPT)
import logging

def prepare_dataset(config: dict) -> DatasetDict:
    """Load OpenR1-Math-220k dataset and re-format into the columns GRPOTrainer expects."""
    raw_ds = load_dataset(config["dataset"]["name"], "default")
    
    # Since there's only a train split in DeepMath-103K, create train/val splits
    split_ds = raw_ds["train"].train_test_split(test_size=0.1, seed=config["training"]["seed"])
    
    processed = {
        "train": split_ds["train"].map(
            lambda x: _openr1_to_grpo(x), 
            remove_columns=raw_ds["train"].column_names
        ),
        "validation": split_ds["test"].map(
            lambda x: _openr1_to_grpo(x), 
            remove_columns=raw_ds["train"].column_names
        )
    }
    
    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info("Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")

    return DatasetDict(processed)

def _openr1_to_grpo(example: dict) -> dict:
    """Convert an OpenR1-Math-220k row → GRPO expected format.
    
    Args:
        example: An OpenR1-Math-220k example
    """
    question = example["problem"].strip()
    answer = example["answer"].strip()
    
    prompt = SYSTEM_PROMPT + "\nUser:" + question + "\nAssistant:"
    return {
        "prompt": prompt,
        "answer": answer,
    }
