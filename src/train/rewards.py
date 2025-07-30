"""Reward functions for GRPO training."""
import re
from typing import Optional, List, Union

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions: List[Union[str, List[dict]]], answer: List[str], **kwargs) -> List[Optional[float]]:
    """
    Reward function that checks for:
      - Exact match / Latex Verified => 1.0
      - Otherwise => 0.0
    """
    full_reward = 1.0
    zero_reward = 0.0

    # Handle both string completions (base models) and chat completions (instruction-tuned models)
    contents = []
    for completion in completions:
        if isinstance(completion, str):
            contents.append(completion)
        else:
            # For chat format (list of dicts)
            contents.append(completion[0]["content"])
    
    rewards: List[Optional[float]] = []
    for comp, sol in zip(contents, answer):
        # extract text
        text = comp.get("content", comp) if isinstance(comp, dict) else comp
        
        # count all answer blocks
        all_ans = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
        if len(all_ans) != 1:
            # zero reward if there are none or more than one
            rewards.append(zero_reward)
            continue
        
        # now we know there's exactly one <answer>…</answer>, so extract it
        content = all_ans[0][len("<answer>"):-len("</answer>")].strip()
        
        # strip whitespace so leading/trailing newlines/spaces
        content = content.strip()
        sol_str = str(sol).strip()

        # pull out what's in the \boxed{…} - using a better regex to handle nested braces
        boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        m_box = re.search(boxed_pattern, content, re.DOTALL)
        if not m_box:
            # no boxed expression → zero reward
            rewards.append(zero_reward)
            continue
        
        # Extract boxed content
        boxed_content = m_box.group(1).strip()
        
        # 0) Direct string comparison for simple string answers like "C", "Yes", etc.
        if boxed_content.lower() == sol_str.lower():
            rewards.append(full_reward)
            continue
        
        # 1) try plain numeric comparison
        try:
            if float(boxed_content) == float(sol_str):
                rewards.append(full_reward)
                continue
        except ValueError:
            # e.g. boxed_content == '84/2'
            pass

        # 2) fallback to symbolic equivalence check
        sol_str = f'${sol_str}$' # add $ to make it a valid latex expression
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
            rewards.append(zero_reward)
            continue
        try:
            if verify(sol_parsed, content_parsed):
                # if the verification passes full reward
                rewards.append(full_reward)
            else:
                # if the verification fails, the content is not correct
                rewards.append(zero_reward)
        except Exception as e:
            # if the verification fails
            rewards.append(zero_reward)

    return rewards


def format_reward(completions: List[Union[str, dict, List[dict]]], **kwargs) -> List[float]:
    """
    Reward function that checks for:
      - Exactly one <think>…</think> and one <answer>…</answer>.
      - Full match: <think>…</think> immediately followed by <answer>…\\boxed{…}…</answer>
      - Partial match: <think>…</think> immediately followed by <answer>…</answer> (no \\boxed)
      - Otherwise => Zero reward
    """
    full_reward = 1.0
    partial_reward = 0.5
    zero_reward = 0.0

    full_pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?\\boxed\{.*?\}.*?</answer>$",
        re.DOTALL
    )
    partial_pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?</answer>$",
        re.DOTALL
    )

    # Handle both string completions (base models) and chat completions (instruction-tuned models)
    contents = []
    for completion in completions:
        if isinstance(completion, str):
            contents.append(completion)
        elif isinstance(completion, list):
            # For chat format (list of dicts)
            contents.append(completion[0]["content"])
        else:
            # For direct dict format
            contents.append(completion.get("content", completion))

    rewards: List[float] = []
    for content in contents:
        # support dict-based or plain-string completions
        text = content.get("content", content) if isinstance(content, dict) else content

        # 1) must have exactly one <think>…</think>
        all_thinks = re.findall(r"<think>.*?</think>", text, re.DOTALL)
        if len(all_thinks) != 1:
            rewards.append(zero_reward)
            continue

        # 2) must have exactly one <answer>…</answer>
        all_answers = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
        if len(all_answers) != 1:
            rewards.append(zero_reward)
            continue

        # 3) strip and apply patterns
        stripped = text.strip()
        if full_pattern.match(stripped):
            rewards.append(full_reward)
        elif partial_pattern.match(stripped):
            rewards.append(partial_reward)
        else:
            rewards.append(zero_reward)

    return rewards


def latent_format_reward(completions: List[Union[str, dict, List[dict]]], **kwargs) -> List[float]:
    """
    Reward function that checks for:
      - Exactly one <|start-latent|>…<|end-latent|> and <answer>…</answer> blocks.
      - Full match: <|start-latent|>…<|end-latent|> <answer>…\\boxed{…}…</answer>
      - Partial match: <|start-latent|>…<|end-latent|> <answer>…</answer> (no \\boxed)
      - Negative reward: any of <|start-latent|> <|end-latent|> <|latent|> tokens are present after the first <|end-latent|> token
      - Otherwise => Zero reward
    """
    full_reward = 1.0
    partial_reward = 0.5
    negative_reward = -1.0
    zero_reward = 0.0

    full_pattern = re.compile(
        r"^<\|start-latent\|>.*?<\|end-latent\|>\s*<answer>\\boxed\{.*?\}</answer>$",
        re.DOTALL
    )
    partial_pattern = re.compile(
        r"^<\|start-latent\|>.*?<\|end-latent\|>\s*<answer>.*?</answer>$",
        re.DOTALL
    )

    # Handle both string completions (base models) and chat completions (instruction-tuned models)
    contents = []
    for completion in completions:
        if isinstance(completion, str):
            contents.append(completion)
        elif isinstance(completion, list):
            # For chat format (list of dicts)
            contents.append(completion[0]["content"])
        else:
            # For direct dict format
            contents.append(completion.get("content", completion))

    rewards: List[float] = []
    for content in contents:
        # support dict-based or plain-string completions
        text = content.get("content", content) if isinstance(content, dict) else content

        # 0) check for negative reward
        # if any of <|start-latent|> <|end-latent|> <|latent|> tokens are present after the first <|end-latent|> token
        first_end_idx = text.find("<|end-latent|>")
        if first_end_idx != -1:
            remainder = text[first_end_idx + len("<|end-latent|>") :]
            if re.search(r"<\|(start-latent|end-latent|latent)\|>", remainder):
                rewards.append(negative_reward)
                continue  # skip further checks for this completion        

        # 1) must have exactly one <|start-latent|>..<|end-latent|>
        all_latent_thinks = re.findall(r"<\|start-latent\|>.*?<\|end-latent\|>", text, re.DOTALL)
        if len(all_latent_thinks) != 1:
            rewards.append(zero_reward)
            continue

        # # 2) must have exactly one <think>…</think>
        # all_thinks = re.findall(r"<think>.*?</think>", text, re.DOTALL)
        # if len(all_thinks) != 1:
        #     rewards.append(zero_reward)
        #     continue

        # 3) must have exactly one <answer>..</answer>
        all_answers = re.findall(r"<answer>.*?</answer>", text, re.DOTALL)
        if len(all_answers) != 1:
            rewards.append(zero_reward)
            continue

        # 3) strip and apply patterns
        stripped = text.strip()
        if full_pattern.match(stripped):
            rewards.append(full_reward)
        elif partial_pattern.match(stripped):
            rewards.append(partial_reward)
        else:
            rewards.append(zero_reward)

    return rewards

if __name__ == "__main__":
    # Example
    # === Example: Symbolic expression ===
    RESPONSE2 = r"""
    <|start-latent|><|latent|><|latent|><|latent|><|end-latent|>
    <think>
    here is the reasoning in language space
    </think>
    <answer>
    sth here is the answer \\boxed{-\dfrac{1}{4}}
    </answer>
    """

    # Ground-truth answer for accuracy_reward:
    ANSWER2 = r"-\dfrac{1}{4}"

    # Run the checks
    for i, (resp, ans) in enumerate([(RESPONSE2, ANSWER2)], start=1):
        f_r = latent_format_reward(completions=[resp])
        a_r = accuracy_reward(prompts=[""], completions=[resp], answer=[ans])
        print(f"Example {i} format_reward: {f_r}")
        print(f"Example {i} accuracy_reward: {a_r}")