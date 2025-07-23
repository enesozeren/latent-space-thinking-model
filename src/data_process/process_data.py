import logging
from typing import Optional
import re
import torch
import math
from datasets import load_dataset, DatasetDict
import random

from prompts.prompts import (
    SYSTEM_PROMPT, 
    SYSTEM_PROMPT_LATENT_REASONER_THINK,
    SYSTEM_PROMPT_LATENT_REASONER_NO_THINK
)
from src.train.utils import count_max_total_latents


def _openr1_to_grpo(example: dict, is_latent_reasoner: bool) -> dict:
    """Convert an OpenR1-Math-220k row → GRPO expected format.
    
    Args:
        example: An OpenR1-Math-220k example
    """
    question = example["problem"].strip()
    answer = example["answer"].strip()
    system_prompt = SYSTEM_PROMPT if not is_latent_reasoner else SYSTEM_PROMPT_LATENT_REASONER_NO_THINK
    prompt = system_prompt + "\nUser:" + question + "\nAssistant:"

    return {
        "prompt": prompt,
        "answer": answer,
    }


def prepare_dataset_rl(config: dict, is_latent_reasoner: bool = False) -> DatasetDict:
    """Load OpenR1-Math-220k dataset and re-format into the columns GRPOTrainer expects."""
    raw_ds = load_dataset(config["dataset"]["name"], "extended", split="train")
    # select the last num_examples_from_last examples
    raw_ds = raw_ds.select(range(config["dataset"]["num_examples"]))
    
    split_ds = raw_ds.train_test_split(test_size=0.025, seed=config["training"]["seed"])
    
    processed = {
        "train": split_ds["train"].map(
            lambda x: _openr1_to_grpo(x, is_latent_reasoner), 
            remove_columns=raw_ds.column_names
        ),
        "validation": split_ds["test"].map(
            lambda x: _openr1_to_grpo(x, is_latent_reasoner), 
            remove_columns=raw_ds.column_names
        )
    }
    
    # Log the dataset split sizes
    logger = logging.getLogger(__name__)
    logger.info(f"Is LatR?: {is_latent_reasoner}")    
    logger.info("Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")

    return DatasetDict(processed)


def _gsm8k_aug_nl_to_sft(
    example, 
    tokenizer, 
    is_latent_reasoner: bool,
    num_latent_per_step: int,
    max_num_latents: int,
    total_num_latents: int) -> dict:
    """Process a single whynlp/gsm8k-aug-nl example into SFT format with explicit <think>/<answer> sections.
    Returns a dict with input_ids, attention_mask and labels.
    """
    question = example["question"].strip()
    think_steps_list = example["steps"]
    answer_text = example["answer"].strip()

    final_ans = f"\\boxed{{{answer_text}}}"
    final_ans = "<answer>" + final_ans + "</answer>"

    # build token sequence
    answer_ids = tokenizer(final_ans, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id

    # latent-token replacement logic
    if is_latent_reasoner:
        # latent token ids
        start_latent_id = tokenizer.start_latent_token_id
        end_latent_id = tokenizer.end_latent_token_id
        latent_id = tokenizer.latent_token_id
        # Calculate the number of latent steps for each example
        num_latent_steps_in_think_block = int(min(
            total_num_latents, 
            len(think_steps_list) * num_latent_per_step
        ))
        # Reduce the tokens in the think block
        if num_latent_steps_in_think_block < max_num_latents:
            think_steps_list = think_steps_list[int(num_latent_steps_in_think_block // num_latent_per_step):]
        else:
            think_steps_list = []
        # prefix
        if think_steps_list == []:
            # no language steps left, so use prompt without think tokens
            system_prompt = SYSTEM_PROMPT_LATENT_REASONER_NO_THINK
        else:
            # some language steps left, so use prompt with think tokens
            system_prompt = SYSTEM_PROMPT_LATENT_REASONER_THINK
        prefix_text = system_prompt + "\nUser:" + question + "\nAssistant:"
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids            
        # Add start think and end think tokens if necessary
        if think_steps_list == []:
            think_ids = []
        else:
            think_steps_str = " ".join(think_steps_list)
            think_text = "<think>" + think_steps_str + "</think>"
            think_ids = tokenizer(think_text, add_special_tokens=False).input_ids

        # Create input_ids
        input_ids = prefix_ids + \
            [start_latent_id] + [latent_id] * num_latent_steps_in_think_block + [end_latent_id] + \
            think_ids + answer_ids + [eos_id]
        # Create labels
        labels = [-100] * len(prefix_ids) + \
            [start_latent_id] + [-100] * num_latent_steps_in_think_block + [end_latent_id] + \
            think_ids + answer_ids + [eos_id]
    else:
        # prefix
        system_prompt = SYSTEM_PROMPT
        prefix_text   = system_prompt + "\nUser:" + question + "\nAssistant:"
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        # think part  
        think_steps_str = " ".join(think_steps_list)
        # convert the think text to ids after adding the start and end think tokens
        think_text = "<think>" + think_steps_str + "</think>"
        think_ids = tokenizer(think_text, add_special_tokens=False).input_ids
        # Create input_ids
        input_ids = prefix_ids + think_ids + answer_ids + [eos_id]
        # Create labels
        labels = [-100] * len(prefix_ids) + think_ids + answer_ids + [eos_id]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels
    }


def prepare_dataset_sft(dataset_name, num_examples, tokenizer, seed: int, 
                        is_latent_reasoner: bool,
                        num_latent_per_step: int,
                        max_num_latents: int,
                        total_num_latents: int) -> DatasetDict:
    """
    Convert whynlp/gsm8k-aug-nl into SFT format using the provided tokenizer.
    """
    logger = logging.getLogger(__name__)
    # Load dataset
    raw_ds = load_dataset(dataset_name, split="train")
    raw_ds = raw_ds.select(range(num_examples))

    # Create train/validation splits
    split_ds = raw_ds.train_test_split(test_size=0.025, seed=seed)

    # Process both splits with map, passing tokenizer & num_latent_steps via fn_kwargs
    processed = {
        "train": split_ds["train"].map(
            _gsm8k_aug_nl_to_sft,
            num_proc=8,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "is_latent_reasoner": is_latent_reasoner,
                "num_latent_per_step": num_latent_per_step,
                "max_num_latents": max_num_latents,
                "total_num_latents": total_num_latents
                },
            remove_columns=raw_ds.column_names,
        ),
        "validation": split_ds["test"].map(
            _gsm8k_aug_nl_to_sft,
            num_proc=8,        
            fn_kwargs={
                "tokenizer": tokenizer, 
                "is_latent_reasoner": is_latent_reasoner,
                "num_latent_per_step": num_latent_per_step,
                "max_num_latents": max_num_latents,
                "total_num_latents": total_num_latents
                },
            remove_columns=raw_ds.column_names,
        ),
    }


    # Log the dataset split sizes
    logger.info("SFT Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")

    return DatasetDict(processed)
