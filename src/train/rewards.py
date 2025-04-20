import re
from typing import List, Optional

# Match answers in the format <answer> any text \boxed{answer} any text </answer>
ANSWER_PATTERN = re.compile(r"<answer>.*?\\boxed\{(.*?)\}.*?</answer>", re.DOTALL)

def accuracy_reward(*, prompts: List[str], completions: List[str], answer: List[str]) -> List[Optional[float]]:
    rewards: List[Optional[float]] = []

    for comp, gold in zip(completions, answer):
        # Support both dict-based and string-based completions
        content = comp.get("content", comp) if isinstance(comp, dict) else comp
        
        m = ANSWER_PATTERN.search(content)
        if m is None: # model gave no answer in the expected format
            rewards.append(0)
            continue

        pred = m.group(1).strip()
        gold = str(gold).strip()

        rewards.append(1.0 if pred == gold else 0.0)

    return rewards


def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function that checks for:
      - Full match: <think>…</think> and <answer>…\boxed{…}…</answer>  => 1.0
      - Partial match: <think>…</think> and <answer>…</answer> (no \\boxed) => 0.5
      - Otherwise => 0.0
    """
    full_pattern    = re.compile(r"^<think>.*?</think>.*?<answer>.*?\\boxed\{.*?\}.*?</answer>$", re.DOTALL)
    partial_pattern = re.compile(r"^<think>.*?</think>.*?<answer>.*?</answer>$", re.DOTALL)

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
# ANSWER="""
# <think> Let Kilee's current age be $x$. We are given that Kilee is currently 20 years old, so $x=20$.
# In 10 years, Kilee's age will be $x+10$, and Cornelia's age will be $3(x+10)$.
# We are given that in 10 years, Cornelia will be three times as old as Kilee. Thus, we have the equation:
# $3(x+10) = 3(x)$
# $3x+30 = 3x$
# $30 = 0$
# This is a contradiction. The initial assumption that the question is well-posed is not valid. </think> <answer> \\boxed{30} </answer>
# """
# print(ANSWER)
# f_reward = format_reward(completions=[ANSWER])
# print("format reward:", f_reward)
# a_reward = accuracy_reward(prompts=[ANSWER], completions=[ANSWER], answer=[30])
# print("accuracy reward:", a_reward)
