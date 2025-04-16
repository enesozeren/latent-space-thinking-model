SYSTEM_PROMPT="""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> </think> and
<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> \\boxed{{}} </answer>. 
User: {user_input}
Assistant: <think>
"""

# Example usage:
# print(SYSTEM_PROMPT.format(user_input="What is the sum of 1 and 2?"))