SYSTEM_PROMPT=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
i.e., <think> reasoning process here </think> <answer> \\boxed{...} </answer>
"""

SYSTEM_PROMPT_LATENT_REASONER=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <|start-latent|> <|end-latent|> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
i.e., <|start-latent|> reasoning process here <|end-latent|> <answer> \\boxed{...} </answer>
"""

SYSTEM_PROMPT_GSM8K_1_SHOT_EVAL=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.

Example 1:
User: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Assistant: <think>
Weng earns 12 / 60 = 0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = 10.
</think>
<answer> \\boxed{10} </answer>

Example 2:
"""

SYSTEM_PROMPT_GSM8K_4_SHOT_EVAL=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.

Example 1:
User: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Assistant: <think>
Weng earns 12 / 60 = 0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = 10.
</think>
<answer> \\boxed{10} </answer>

Example 2:
User: Bobby has 16 toy cars, and the number of cars he has increases by 50% every year. How many toy cars will Bobby have in three years?
Assistant: <think>
In the first year, Bobby will acquire 16 * .5 = 8 new cars.
After the first year, he will have a total of 16 + 8 = 24 cars.
In the second year, Bobby will acquire 24 * .5 = 12 new cars.
After the second year, he will have 24 + 12 = 36 cars in total.
In the third year, Bobby will acquire 36 * .5 = 18 new cars.
After the third year, he will have 36 + 18 = 54 cars in total.
</think>
<answer> \\boxed{54} </answer>

Example 3:
User: James was 2/3s as tall as his uncle who is 72 inches. He gets a growth spurt that makes him gain 10 inches. How much taller is his uncle than James now?
Assistant: <think>
He was 72*2/3=48 inches tall before.
He is now 48+10=58 inches.
So his uncle is 72-58=14 inches taller.
</think>
<answer> \\boxed{14} </answer>

Example 4:
User: Arabella is a dance student learning three new steps this session. Her instructor has her spend thirty minutes on learning the first step. The second step she masters in half the time. The third step is more complex, so it takes her as long as both the other steps to learn. How many minutes did she spend learning the three steps?
Assistant: <think>
Arabella spent 30 / 2 = 15 minutes on the second step.
She took 30 + 15 = 45 minutes to learn the third step.
Therefore, Arabella spent 30 + 15 + 45 = 90 minutes learning all three steps.
</think>
<answer> \\boxed{90} </answer>

Example 5:
"""

SYSTEM_PROMPT_MATH500_1_SHOT_EVAL=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.

Example 1:
User: How many positive whole-number divisors does 196 have?
Assistant: <think>
First prime factorize $196=2^2\cdot7^2$. The prime factorization of any divisor of 196 cannot include any primes other than 2 and 7. 
We are free to choose either 0, 1, or 2 as the exponent of 2 in the prime factorization of a divisor of 196. 
Similarly, we may choose 0, 1, or 2 as the exponent of 7. In total, there are $3\times 3=9$ possibilities for the prime factorization of a divisor of 196. 
Distinct prime factorizations correspond to distinct integers, so there are 9 divisors of 196.
</think>
<answer> \\boxed{9} </answer>

Example 2:
"""

# # Example usage
# ex_prompt = SYSTEM_PROMPT_MATH500_1_SHOT_EVAL + "User: What is the sum of 1 and 2?" + "\nAssistant:"
# print(ex_prompt)