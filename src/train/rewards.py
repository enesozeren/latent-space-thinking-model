"""Reward functions for GRPO training."""
import re
from typing import Optional, List, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(*, prompts: List[str], completions: List[str], answer: List[str]) -> List[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth"""
    
    contents = [completion[0]["content"] for completion in completions]
    
    rewards: List[Optional[float]] = []
    for comp, sol in zip(contents, answer):
        # extract text
        text = comp.get("content", comp) if isinstance(comp, dict) else comp
        
        # count all answer blocks
        all_ans = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
        if len(all_ans) != 1:
            # zero reward if there are none or more than one
            rewards.append(0.0)
            continue
        
        # now we know there's exactly one <answer>…</answer>, so extract it
        content = all_ans[0][len("<answer>"):-len("</answer>")].strip()
        
        # strip whitespace so leading/trailing newlines/spaces
        content = content.strip()
        sol_str = str(sol).strip()

        # pull out what's in the \boxed{…}
        m_box = re.search(r"\\boxed\{(.*?)\}", content, re.DOTALL)
        if not m_box:
            # no boxed expression → zero reward
            rewards.append(0.0)
            continue
        
        # 1) try plain numeric comparison
        boxed_content = m_box.group(1).strip()
        try:
            if float(boxed_content) == float(sol_str):
                rewards.append(1.0)
                continue
        except ValueError:
            # e.g. boxed_content == '84/2'
            pass

        # 2) fallback to symbolic equivalence check
        sol_parsed = parse(sol_str, extraction_mode="first_match")
        if sol_parsed is None:
            rewards.append(None)
            continue

        content_parsed = parse(
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
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if content_parsed is None:
            # if the content is not parseable
            rewards.append(0.0)
            continue
        try:
            if verify(sol_parsed, content_parsed):
                # if the verification passes, we can verify it but it is not perfect match
                rewards.append(0.5)
            else:
                # if the verification fails, the content is not correct
                rewards.append(0.0)
        except Exception as e:
            # if the verification fails
            rewards.append(0.0)

    return rewards


def format_reward(completions: List[Union[str, dict]], **kwargs) -> List[float]:
    """
    Reward function that checks for:
      - Exactly one <think>…</think> and one <answer>…</answer>.
      - Full match: <think>…</think> immediately followed by <answer>…\\boxed{…}…</answer>  => 1.0
      - Partial match: <think>…</think> immediately followed by <answer>…</answer> (no \\boxed) => 0.5
      - Otherwise => 0.0
    """
    full_pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?\\boxed\{.*?\}.*?</answer>$",
        re.DOTALL
    )
    partial_pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?</answer>$",
        re.DOTALL
    )

    contents = [completion[0]["content"] for completion in completions]
    rewards: List[float] = []
    for comp in contents:
        # support dict-based or plain-string completions
        content = comp.get("content", comp) if isinstance(comp, dict) else comp

        # 1) must have exactly one <think>…</think>
        all_thinks = re.findall(r"<think>.*?</think>", content, re.DOTALL)
        if len(all_thinks) != 1:
            rewards.append(0.0)
            continue

        # 2) must have exactly one <answer>…</answer>
        all_answers = re.findall(r"<answer>.*?</answer>", content, re.DOTALL)
        if len(all_answers) != 1:
            rewards.append(0.0)
            continue

        # 3) strip and apply patterns
        stripped = content.strip()
        if full_pattern.match(stripped):
            rewards.append(1.0)
        elif partial_pattern.match(stripped):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards

# # Example
# ANSWER="""
# <think> reasoning </think>
# <answer> result is \\boxed{42} </answer>
# """
# print(ANSWER)
# f_reward = format_reward(completions=[ANSWER])
# print("format reward:", f_reward)
# a_reward = accuracy_reward(prompts=[""], completions=[ANSWER], answer=["42"])
# print("accuracy reward:", a_reward)
