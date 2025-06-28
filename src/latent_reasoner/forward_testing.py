import torch
from src.latent_reasoner.model import LatentReasoner
from src.train.utils import setup_latent_tokens
from src.data_process.process_data import _openr1math_to_sft
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name_or_path = "Qwen/Qwen2.5-0.5B"
# Create the model
model = LatentReasoner.from_pretrained(model_name_or_path).to(device)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Define new tokens for latent steps and think/answer
model, tokenizer = setup_latent_tokens(model, tokenizer, is_latent_reasoner=True)

example = {
    "problem": "An odd number is multiplied by the two odd numbers adjacent to it, resulting in two products that differ by 44. What is this odd number?",
    "solution": "the two odd numbers adjacent to it can be represented as $x-2$ and $x+2$. According to the problem, we can establish the equation:",
    "answer": "11"}

example_input = _openr1math_to_sft(
    example, 
    tokenizer, 
    num_tokens_per_latent=5, 
    add_num_latents_per_update=1, 
    update_cycle=2,
    is_latent_reasoner=True
)

# Convert input to tensors and move to device by adding the batch dimension
example_input = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in example_input.items()}

# Forward pass through the model
with torch.no_grad():
    output = model.sft_forward(**example_input)
    logits = output.logits
