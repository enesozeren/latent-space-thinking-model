SYSTEM_PROMPT=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \boxed{...}.
Output only those two tagged blocks, in that order, nothing else.
i.e., <think> reasoning process here </think> <answer> \boxed{...} </answer>
"""

SYSTEM_PROMPT_LATENT_REASONER=r"""
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. 
The assistant first thinks about the reasoning process in the latent-space without using language.
Then it provides the user with the reasoning in latent-space, reasoning in language-space and the answer. 
The reasoning in latent-space, in language-space and answer are enclosed within <|start-latent|> <|end-latent|>, <think> </think> and <answer> </answer> tags, respectively and the answer is boxed using \boxed{...}.
Output only those three tagged blocks, in that order, nothing else.
i.e., <|start-latent|> reasoning in latent here <|end-latent|> <think> reasoning in language here </think> <answer> \boxed{...} </answer>
"""

GSM8K_1_SHOT_EVAL=r"""
Example 1:
User:Alex earns $9 an hour for cleaning a home. Yesterday, he did 90 minutes of cleaning. How much did hhe earn?
Assistant:<think>Alex earns $9 per hour for cleaning.
He worked 90 minutes which is 90/60 = 1.5 hours.
Working 90 minutes, he earned 9 x 1.5 = 13.5</think><answer>\boxed{13.5}</answer>

Example 2:
"""

SYSTEM_PROMPT_GSM8K_1_SHOT_EVAL = SYSTEM_PROMPT + GSM8K_1_SHOT_EVAL
SYSTEM_PROMPT_LATENT_REASONER_GSM8K_1_SHOT_EVAL = SYSTEM_PROMPT_LATENT_REASONER + GSM8K_1_SHOT_EVAL

MATH500_1_SHOT_EVAL=r"""
Example 1:
User:How many different positive integer divisors does 48 have?
Assistant:<think>First prime factorize $48=2^4\cdot3^1$. The prime factorization of any divisor of 48 can only include the primes 2 and 3. 
We are free to choose either 0, 1, 2, 3 or 4 as the exponent of 2 (5 options).
For the exponent of 3, we can choose 0 or 1 (2 options).
In total, there are $5\times 2=10$ possibilities.</think><answer>\boxed{10}</answer>

Example 2:
"""

SYSTEM_PROMPT_MATH500_1_SHOT_EVAL = SYSTEM_PROMPT + MATH500_1_SHOT_EVAL
SYSTEM_PROMPT_LATENT_REASONER_MATH500_1_SHOT_EVAL = SYSTEM_PROMPT_LATENT_REASONER + MATH500_1_SHOT_EVAL

# # Example usage
# ex_prompt = SYSTEM_PROMPT_LATENT_REASONER_MATH500_1_SHOT_EVAL + "User: What is the sum of 1 and 2?" + "\nAssistant:"
# print(ex_prompt)