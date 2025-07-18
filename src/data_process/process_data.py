import logging
from typing import Optional
import re
import torch
import math
from datasets import load_dataset, DatasetDict
import random

from prompts.prompts import (
    SYSTEM_PROMPT, 
    SYSTEM_PROMPT_LATENT_REASONER
)
from src.train.utils import count_max_total_latents

ANSWER_PATTERN = re.compile(r'The answer is:\s*([^\n\r#]+)', flags=re.IGNORECASE)

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


def _metamathqa_to_sft(
    example, 
    tokenizer, 
    is_latent_reasoner: bool,
    num_tokens_per_latent: int,
    max_num_latents: int,
    total_num_latents: int) -> dict:
    """Process a single meta-math/MetaMathQA example into SFT format with explicit <think>/<answer> sections.
    If num_tokens_per_latent is provided, it will be used to replace the language tokens with latent steps.
    Returns a dict with input_ids, attention_mask and labels.
    """
    question = example["query"].strip()
    think_text = example["response"].strip()
    # parse the answer from the think text in meta-math/MetaMathQA dataset
    # response always contains the answer is at the end
    m = ANSWER_PATTERN.search(think_text)
    answer_found = m is not None
    answer_text = m.group(1).strip() if m else "answer not found"

    final_ans = f"\\boxed{{{answer_text}}}"

    # add system prompt,<think> and <answer> tags to the COT and final answer
    system_prompt = SYSTEM_PROMPT if not is_latent_reasoner else SYSTEM_PROMPT_LATENT_REASONER
    prefix_text   = system_prompt + "\nUser:" + question + "\nAssistant:"    
    final_ans = "<answer>" + final_ans + "</answer>"

    # build token sequence
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(final_ans, add_special_tokens=False).input_ids
    eos_id = tokenizer.eos_token_id
    if is_latent_reasoner:
        start_latent_id = tokenizer.start_latent_token_id
        end_latent_id = tokenizer.end_latent_token_id
        latent_id = tokenizer.latent_token_id

    # latent-token replacement logic
    if is_latent_reasoner:
        # convert the think text to ids
        think_ids = tokenizer(think_text, add_special_tokens=False).input_ids
        # Random Removal Smoothing (Deng et al., 2024) - We add one more latent for 2% of the time
        if random.randint(1, 100) > 99:
            total_num_latents = total_num_latents + 1
        # Calculate the number of latent steps for each example
        num_latent_steps_in_think_block = min(
            total_num_latents, 
            math.ceil(len(think_ids) // num_tokens_per_latent)
        )
        # Reduce the tokens in the think block
        if num_latent_steps_in_think_block < max_num_latents:
            think_ids = think_ids[num_tokens_per_latent * num_latent_steps_in_think_block:]
        else:
            think_ids = []
        # Add start think and end think tokens in the think ids
        ## first convert the think ids to a string
        think_text = tokenizer.decode(think_ids, skip_special_tokens=False)
        ## then add the start think and end think tokens
        think_text = "<think>" + think_text + "</think>"
        ## then convert the think text back to ids
        think_ids = tokenizer(think_text, add_special_tokens=False).input_ids

        # Create the input_ids
        input_ids = prefix_ids + \
            [start_latent_id] + [latent_id] * num_latent_steps_in_think_block + [end_latent_id] + \
            think_ids + answer_ids + [eos_id]
        # Mask prefix and latent tokens but not the think/answer sections
        labels = [-100] * len(prefix_ids) + \
            [start_latent_id] + [-100] * num_latent_steps_in_think_block + [end_latent_id] + \
            think_ids + answer_ids + [eos_id]
    else:
        # convert the think text to ids after adding the start and end think tokens
        think_text = "<think>" + think_text + "</think>"
        think_ids = tokenizer(think_text, add_special_tokens=False).input_ids
        # No latent tokens
        input_ids = prefix_ids + think_ids + answer_ids + [eos_id]
        # Only mask the prefix
        labels = [-100] * len(prefix_ids) + think_ids + answer_ids + [eos_id]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "answer_found":   answer_found
    }


def prepare_dataset_sft(dataset_name, num_examples, tokenizer, seed: int, 
                        is_latent_reasoner: bool,
                        num_tokens_per_latent: int,
                        max_num_latents: int,
                        total_num_latents: int) -> DatasetDict:
    """Convert meta-math/MetaMathQA into latent-reasoning SFT format using the provided tokenizer.
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
            _metamathqa_to_sft,
            num_proc=4,
            fn_kwargs={
                "tokenizer": tokenizer, 
                "is_latent_reasoner": is_latent_reasoner,
                "num_tokens_per_latent": num_tokens_per_latent,
                "max_num_latents": max_num_latents,
                "total_num_latents": total_num_latents
                },
            remove_columns=raw_ds.column_names,
        ),
        "validation": split_ds["test"].map(
            _metamathqa_to_sft,
            num_proc=4,        
            fn_kwargs={
                "tokenizer": tokenizer, 
                "is_latent_reasoner": is_latent_reasoner,
                "num_tokens_per_latent": num_tokens_per_latent,
                "max_num_latents": max_num_latents,
                "total_num_latents": total_num_latents
                },
            remove_columns=raw_ds.column_names,
        ),
    }

    # Drop examples where the answer was missing
    processed["train"]      = processed["train"].filter(lambda ex: ex["answer_found"])
    processed["validation"] = processed["validation"].filter(lambda ex: ex["answer_found"])

    # (optional) tidy up
    processed["train"]      = processed["train"].remove_columns("answer_found")
    processed["validation"] = processed["validation"].remove_columns("answer_found")


    # Log the dataset split sizes
    logger.info("SFT Dataset split sizes:")
    logger.info(f"  Train: {len(processed['train'])} examples")
    logger.info(f"  Validation: {len(processed['validation'])} examples")

    return DatasetDict(processed)
