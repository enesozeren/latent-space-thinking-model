"""Reward functions for GRPO training."""
import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional, List

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(*, prompts: List[str], completions: List[str], answer: List[str]) -> List[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    for content, sol in zip(completions, answer):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks for:
      - Full match: <think>…</think> and <answer>…\boxed{…}…</answer>  => 1.0
      - Partial match: <think>…</think> and <answer>…</answer> (no \\boxed) => 0.5
      - Otherwise => 0.0
    
    The function ensures that only whitespace (spaces, tabs, newlines) can appear 
    between </think> and <answer> tags.
    """
    full_pattern    = re.compile(r"^<think>.*?</think>\s*<answer>.*?\\boxed\{.*?\}.*?</answer>$", re.DOTALL)
    partial_pattern = re.compile(r"^<think>.*?</think>\s*<answer>.*?</answer>$", re.DOTALL)

    rewards: List[float] = []
    for comp in completions:
        # support dict-based or plain-string completions
        content = comp.get("content", comp) if isinstance(comp, dict) else comp

        # strip whitespace so leading/trailing newlines/spaces don't block the match
        content = content.strip()

        if full_pattern.match(content):
            rewards.append(1.0)
        elif partial_pattern.match(content):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards

# # Example
# ANSWER="""<think>
# First, I consider the inputs and check the edge cases.
# Then, I work through the main logic step by step.</think>
# <answer>
# \\boxed{84/2}
# </answer>
# """
# print(ANSWER)
# f_reward = format_reward(completions=[ANSWER])
# print("format reward:", f_reward)
# a_reward = accuracy_reward(completions=[ANSWER], answers=["42"])
# print("accuracy reward:", a_reward)
