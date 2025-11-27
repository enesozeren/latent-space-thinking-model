import torch
from transformers import AutoTokenizer
import argparse

from src.latent_reasoner.model import LatentReasoner
from src.train_sft_grpo.utils import setup_latent_tokens
from prompts.prompts import (
    SYSTEM_PROMPT_LATENT_REASONER_NO_THINK
)

def main(model_name_or_path, question):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the model
    model = LatentReasoner.from_pretrained(model_name_or_path).to(device)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model, tokenizer = setup_latent_tokens(
        model=model, 
        tokenizer=tokenizer,
        is_latent_reasoner=True
    )
    
    system_prompt = SYSTEM_PROMPT_LATENT_REASONER_NO_THINK
    prompt   = system_prompt + "\nUser:" + question + "\nAssistant:"    

    # Tokenize batch of prompts
    tokenized = tokenizer(prompt, return_tensors="pt", padding=True)
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    # Print input shapes
    print(f"Prompt:\n{prompt}")
    print(f"Prompt IDs Shape: {prompt_ids.shape}")

    # Generate responses for the batch
    completion_token_ids, prompt_completion_embeds = model.generate(
        prompt_ids,
        attention_mask=attention_mask,
        num_latent_steps=6,  # Set the number of latent steps        
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        generation_config=None
    )

    # Print the prompt_completion_embeds shape
    print(f"Prompt Completion Embeds Shape: {prompt_completion_embeds.shape}")
    # Print the completion_token_ids shape
    print(f"Completion Token IDs Shape: {completion_token_ids.shape}")
    
    # Decode the generated response
    response_text = tokenizer.decode(completion_token_ids[0], skip_special_tokens=False)
    
    # Print the prompt and response
    print(f"Prompt: {prompt!r}")
    print(f"Response: {response_text!r}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with LatentReasoner model")
    parser.add_argument("--model_name_or_path", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra32qov2/latent_reasoner_storage/outputs/latent_reasoner_1p5b_sft/20250719_210442/final_model",
                       help="Path to the pretrained model or model name")
    parser.add_argument("--question", type=str, 
                       default=r"If James collects 15 gallons of water for every inch of rain, and it rained 4 inches on Monday and 3 inches on Tuesday, how much money did he make from selling all the water if he can sell it for $1.2 per gallon?",
                       help="Question to ask the model")
    
    args = parser.parse_args()
    
    main(args.model_name_or_path, args.question)