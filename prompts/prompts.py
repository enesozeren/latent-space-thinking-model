SYSTEM_PROMPT=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
i.e., <think> reasoning process here </think> <answer> \\boxed{...} </answer>
"""

SYSTEM_PROMPT_LATENT_REASONER=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the latent-space without using language.
Then it provides the user with the reasoning in latent-space, reasoning in language-space and the answer. 
The reasoning in latent-space, in language-space and answer are enclosed within <|start-latent|> <|end-latent|>, <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \\boxed{...}.
Output only those three tagged blocks, in that order, nothing else.
i.e., <|start-latent|> reasoning in latent here <|end-latent|> <think> reasoning in language here </think> <answer> \\boxed{...} </answer>
"""

GSM8K_1_SHOT_EVAL=r"""
Example 1:
User:Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Assistant:<think>Weng earns 12 / 60 = 0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = 10.</think><answer>\\boxed{10}</answer>

Example 2:
"""

SYSTEM_PROMPT_GSM8K_1_SHOT_EVAL = SYSTEM_PROMPT + GSM8K_1_SHOT_EVAL
SYSTEM_PROMPT_LATENT_REASONER_GSM8K_1_SHOT_EVAL = SYSTEM_PROMPT_LATENT_REASONER + GSM8K_1_SHOT_EVAL

MATH500_1_SHOT_EVAL=r"""
Example 1:
User:How many positive whole-number divisors does 196 have?
Assistant: <think>First prime factorize $196=2^2\cdot7^2$. The prime factorization of any divisor of 196 cannot include any primes other than 2 and 7. 
We are free to choose either 0, 1, or 2 as the exponent of 2 in the prime factorization of a divisor of 196. 
Similarly, we may choose 0, 1, or 2 as the exponent of 7. In total, there are $3\times 3=9$ possibilities for the prime factorization of a divisor of 196. 
Distinct prime factorizations correspond to distinct integers, so there are 9 divisors of 196.</think><answer>\\boxed{9}</answer>

Example 2:
"""

SYSTEM_PROMPT_MATH500_1_SHOT_EVAL = SYSTEM_PROMPT + MATH500_1_SHOT_EVAL
SYSTEM_PROMPT_LATENT_REASONER_MATH500_1_SHOT_EVAL = SYSTEM_PROMPT_LATENT_REASONER + MATH500_1_SHOT_EVAL

# # Example usage
# ex_prompt = SYSTEM_PROMPT_LATENT_REASONER_MATH500_1_SHOT_EVAL + "User: What is the sum of 1 and 2?" + "\nAssistant:"
# print(ex_prompt)