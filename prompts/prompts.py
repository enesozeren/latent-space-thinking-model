SYSTEM_PROMPT=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
i.e., <think> reasoning process here </think> <answer> \\boxed{...} </answer>
"""

# # Example usage
# ex_prompt = SYSTEM_PROMPT + "User: What is the sum of 1 and 2?" + "\nAssistant:"
# print(ex_prompt)