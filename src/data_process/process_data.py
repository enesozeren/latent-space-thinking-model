from typing import Optional
import re
import torch
from datasets import load_dataset, DatasetDict
from prompts.prompts import (
    SYSTEM_PROMPT, 
    SYSTEM_PROMPT_LATENT_REASONER
)
import logging


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


def _gsm8k_to_sft(example, tokenizer, max_num_latent_steps: Optional[int] = None) -> dict:
    """Process a single GSM8K example into SFT format and explicit <think>/<answer> sections.
    If `max_num_latent_steps` is provided, it will be used to determine the number of latent steps.
    Returns a dict with input_ids, attention_mask and labels.
    """
    question = example["question"].strip()

    # Convert trailing “####  <value>” to “\boxed{<value>}”
    raw_answer = example.get("answer", "").strip()
    processed = re.sub(r"(?:\n)?####\s*(.+)$", r"\\boxed{\1}", raw_answer)

    # The first \boxed{ … } is taken as the final answer
    m = re.search(r"\\boxed\{[^}]+\}", processed)
    if m:
        cot_text   = processed[:m.start()].rstrip()       # chain-of-thought
        final_ans  = m.group()                            # boxed answer
    else:                                                 # fall-back (shouldn’t happen)
        cot_text, final_ans = processed, ""

    think_block  = f"<think> {cot_text} </think>"
    answer_block = f"<answer> {final_ans} </answer>"
    assistant_response = think_block + "\n" + answer_block

    # build token sequence
    prefix_text   = "\nUser: " + question + "\nAssistant:"
    prefix_ids    = tokenizer(prefix_text, add_special_tokens=False).input_ids
    answer_ids    = tokenizer(assistant_response, add_special_tokens=False).input_ids

    eos_id = tokenizer.eos_token_id
    # latent-token scaffold (only if max_num_latent_steps is provided)
    if max_num_latent_steps is not None:
        num_latent_steps = torch.randint(1, max_num_latent_steps + 1, (1,)).item()
        latent_ids = (
            [tokenizer.start_latent_token_id] +
            [tokenizer.latent_token_id] * num_latent_steps +
            [tokenizer.end_latent_token_id]
        )
        # Add the latent scaffold to the input_ids
        input_ids = prefix_ids + latent_ids + answer_ids + [eos_id]
        # We still train on the whole think+answer block, so only prefix+latent are masked
        labels = [-100] * (len(prefix_ids) + len(latent_ids)) + answer_ids + [eos_id]
    else:
        # No latent tokens - just prefix + answer + eos
        input_ids = prefix_ids + answer_ids + [eos_id]
        # Only mask the prefix, train on the whole think+answer block
        labels = [-100] * len(prefix_ids) + answer_ids + [eos_id]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def prepare_dataset_latent_sft(dataset_name, tokenizer, seed: int, max_num_latent_steps: Optional[int] = None) -> DatasetDict:
    """Convert GSM8K into latent-reasoning SFT format using the provided tokenizer."""
    # Load dataset
    raw_ds = load_dataset(dataset_name, "main", split="train")

    # Create train/validation splits
    split_ds = raw_ds.train_test_split(test_size=0.1, seed=seed)

    # Process both splits with map, passing tokenizer & num_latent_steps via fn_kwargs
    processed = {
        "train": split_ds["train"].map(
            _gsm8k_to_sft,
            fn_kwargs={"tokenizer": tokenizer, "max_num_latent_steps": max_num_latent_steps},
            remove_columns=raw_ds.column_names,
        ),
        "validation": split_ds["test"].map(
            _gsm8k_to_sft,
            fn_kwargs={"tokenizer": tokenizer, "max_num_latent_steps": max_num_latent_steps},
            remove_columns=raw_ds.column_names,
        ),
    }

    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info("GSM8K SFT Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")
    if max_num_latent_steps is not None:
        logger.info(f"  Max number of latent steps: {max_num_latent_steps}")
    else:
        logger.info("  No latent tokens added (max_num_latent_steps is None)")

    return DatasetDict(processed)
