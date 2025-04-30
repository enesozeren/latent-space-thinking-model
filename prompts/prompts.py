SYSTEM_PROMPT=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
i.e., <think> reasoning process here </think> <answer> \\boxed{...} </answer>
"""

SYSTEM_PROMPT_GSM8K_4_SHOT_EVAL=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
Example:

User: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Assistant: <think> 
Weng earns 12 / 60 = 0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = 10.
</think>
<answer> \\boxed{10} </answer>
"""

# # Example usage
# ex_prompt = SYSTEM_PROMPT + "User: What is the sum of 1 and 2?" + "\nAssistant:"
# print(ex_prompt)