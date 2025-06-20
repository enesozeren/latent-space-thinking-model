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
    system_prompt = SYSTEM_PROMPT if not is_latent_reasoner else SYSTEM_PROMPT_LATENT_REASONER
    prompt = system_prompt + "\nUser:" + question + "\nAssistant:"

    return {
        "prompt": prompt,
        "answer": answer,
    }


def prepare_dataset(config: dict, is_latent_reasoner: bool) -> DatasetDict:
    """Load OpenR1-Math-220k dataset and re-format into the columns GRPOTrainer expects."""
    raw_ds = load_dataset(config["dataset"]["name"], "default")
    
    # Since there's only a train split in DeepMath-103K, create train/val splits
    split_ds = raw_ds["train"].train_test_split(test_size=0.025, seed=config["training"]["seed"])
    
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


def _openr1math_to_sft(
    example, 
    tokenizer, 
    is_latent_reasoner: bool,
    num_tokens_per_latent: Optional[int] = None, 
    add_num_latents_per_update: Optional[int] = None,
    update_cycle: int = 0) -> dict:
    """Process a single open-r1/OpenR1-Math-220k example into SFT format with explicit <think>/<answer> sections.
    If num_tokens_per_latent is provided, it will be used to replace the language tokens with latent steps.
    Returns a dict with input_ids, attention_mask and labels.
    """
    question = example["problem"].strip()
    cot_text = example.get("solution", "").strip()
    answer_text = example.get("answer", "").strip()
    final_ans = f"\\boxed{{{answer_text}}}"

    # build token sequence
    system_prompt = SYSTEM_PROMPT if not is_latent_reasoner else SYSTEM_PROMPT_LATENT_REASONER
    prefix_text   = system_prompt + "\nUser:" + question + "\nAssistant:"
    prefix_ids    = tokenizer(prefix_text, add_special_tokens=False).input_ids
    think_ids = tokenizer(cot_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(final_ans, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    start_think_id = tokenizer.start_think_token_id
    end_think_id = tokenizer.end_think_token_id
    start_answer_id = tokenizer.start_answer_token_id
    end_answer_id = tokenizer.end_answer_token_id
    if num_tokens_per_latent and add_num_latents_per_update:
        start_latent_id = tokenizer.start_latent_token_id
        end_latent_id = tokenizer.end_latent_token_id
        latent_id = tokenizer.latent_token_id

    # latent-token replacement logic
    if num_tokens_per_latent and add_num_latents_per_update and update_cycle > 0:
        # Calculate the number of latent steps
        num_latent_steps = min(add_num_latents_per_update * update_cycle, len(think_ids) // num_tokens_per_latent) 
        # Reduce the tokens in the think block
        think_ids = think_ids[num_tokens_per_latent * num_latent_steps:]
        # Create the input_ids
        input_ids = prefix_ids + \
            [start_latent_id] + [latent_id] * num_latent_steps + [end_latent_id] + \
            [start_think_id] + think_ids + [end_think_id] + \
            [start_answer_id] + answer_ids + [end_answer_id] + [eos_id]
        # Mask prefix and latent tokens but not the think/answer sections
        labels = [-100] * len(prefix_ids) + \
            [start_latent_id] + [-100] * num_latent_steps + [end_latent_id] + \
            [start_think_id] + think_ids + [end_think_id] + \
            [start_answer_id] + answer_ids + [end_answer_id] + [eos_id]
    else:
        # No latent tokens
        input_ids = prefix_ids + \
            [start_think_id] + think_ids + [end_think_id] + \
            [start_answer_id] + answer_ids + [end_answer_id] + [eos_id]
        # Only mask the prefix
        labels = [-100] * len(prefix_ids) + \
            [start_think_id] + think_ids + [end_think_id] + \
            [start_answer_id] + answer_ids + [end_answer_id] + [eos_id]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def prepare_dataset_sft(dataset_name, num_examples, tokenizer, seed: int, 
                        is_latent_reasoner: bool,
                        num_tokens_per_latent: Optional[int] = None, 
                        add_num_latents_per_update: Optional[int] = None,
                        update_cycle: int = 0) -> DatasetDict:
    """Convert open-r1/OpenR1-Math-220k into latent-reasoning SFT format using the provided tokenizer."""
    # Load dataset
    raw_ds = load_dataset(dataset_name, "extended", split="train")
    raw_ds = raw_ds.select(range(num_examples))

    # Create train/validation splits
    split_ds = raw_ds.train_test_split(test_size=0.025, seed=seed)

    # Process both splits with map, passing tokenizer & num_latent_steps via fn_kwargs
    processed = {
        "train": split_ds["train"].map(
            _openr1math_to_sft,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "is_latent_reasoner": is_latent_reasoner,
                "num_tokens_per_latent": num_tokens_per_latent,
                "add_num_latents_per_update": add_num_latents_per_update,
                "update_cycle": update_cycle
                },
            remove_columns=raw_ds.column_names,
        ),
        "validation": split_ds["test"].map(
            _openr1math_to_sft,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "is_latent_reasoner": is_latent_reasoner,
                "num_tokens_per_latent": num_tokens_per_latent,
                "add_num_latents_per_update": add_num_latents_per_update,
                "update_cycle": update_cycle
                },
            remove_columns=raw_ds.column_names,
        ),
    }

    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info(f"Update Cycle: {update_cycle}")    
    logger.info("SFT Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")
    if num_tokens_per_latent is not None:
        logger.info(f"  Number of tokens per latent step: {num_tokens_per_latent}")
    else:
        logger.info("  No latent tokens added (num_tokens_per_latent is None)")

    return DatasetDict(processed)
