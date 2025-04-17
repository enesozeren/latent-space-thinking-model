import re
from typing import List, Optional

ANSWER_PATTERN = re.compile(r"<answer>\[(.*?)\]</answer>")

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


def format_reward(completions: List[str], **kwargs):
    """
    Reward function that checks if the completion matches the format:
    <think>...</think><answer>[...]</answer>.
    """
    import re
    # Pattern requires <think>…</think> followed immediately by an <answer> containing [...]
    pattern = r"^<think>.*?</think>\s*<answer>\[.*?\]</answer>$"
    rewards = []
    for comp in completions:
        # Support both dict-based and string-based completions
        content = comp.get("content", comp) if isinstance(comp, dict) else comp
        # Use DOTALL so that newline characters are matched by .*
        match = re.match(pattern, content, re.DOTALL)
        rewards.append(1.0 if match else 0.0)
    return rewards

# # Example
# ANSWER="<think> i am thinkingg </think> <answer>[5]</answer>"
# print(ANSWER)
# f_reward = format_reward(completions=[ANSWER])
# print("format reward:", f_reward)
# a_reward = accuracy_reward(prompts=[ANSWER], completions=[ANSWER], answer=[5])
# print("accuract reward:", a_reward)
