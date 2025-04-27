from datasets import load_dataset, DatasetDict
from prompts.prompts import (SYSTEM_PROMPT)
import logging

def prepare_dataset(config: dict) -> DatasetDict:
    """Load GSM8K and re‑format into the columns GRPOTrainer expects."""
    raw_ds = load_dataset(
        config["dataset"]["name"],
        config["dataset"].get("subname", None),
    )

    # Train/val split
    split_ds = raw_ds["train"].train_test_split(test_size=0.1, seed=config["training"]["seed"])

    # Get model_type if specified, default to "it" (instruction tuned)
    model_type = config["model"].get("model_type", "it")
    
    processed = {
        "train": split_ds["train"].map(
            lambda x: _gsm8k_to_grpo(x, model_type), 
            remove_columns=raw_ds["train"].column_names
        ),
        "validation": split_ds["test"].map(
            lambda x: _gsm8k_to_grpo(x, model_type), 
            remove_columns=raw_ds["train"].column_names
        ),
        "test": raw_ds["test"].map(
            lambda x: _gsm8k_to_grpo(x, model_type), 
            remove_columns=raw_ds["test"].column_names
        ),
    }
    
    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info("Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")
    logger.info(f"  Test: {len(processed['test'])} examples")

    return DatasetDict(processed)

def _gsm8k_to_grpo(example: dict, model_type: str = "it") -> dict:
    """Convert a GSM8K row → GRPO expected format.
    
    Args:
        example: A GSM8K example
        model_type: The model type, either "it" for instruction tuned models (default)
                    or "base" for base models
    """
    question = example["question"].strip()
    answer = _extract_gsm8k_answer(example["answer"])
    
    if model_type == "it":
        # For instruction-tuned models, use the chat template
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return {
            "prompt": prompt,
            "answer": answer,
        }
    elif model_type == "base":
        # For base models, use a simple text prompt without chat template
        prompt = SYSTEM_PROMPT + "\nUser:" + question + "\nAssistant:"
        return {
            "prompt": prompt,
            "answer": answer,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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