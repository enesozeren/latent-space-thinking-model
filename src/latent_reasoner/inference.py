import torch
from src.latent_reasoner.model import LatentReasoner

from transformers import AutoTokenizer

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name_or_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra32qov2/latent_reasoner_storage/outputs/latent_reasoner_1p5b_sft/20250530_151612/checkpoint-52/"
    # Create the model
    model = LatentReasoner.from_pretrained(model_name_or_path).to(device)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Define new tokens for latent steps
    START = "<|start-latent|>"
    LAT   = "<|latent|>"
    END   = "<|end-latent|>"
    new_specials = [START, LAT, END]

    # Only add / resize / init if it is not the first training run
    if not all(tok in tokenizer.get_vocab() for tok in new_specials):
        # 3) Add them to the tokenizer’s vocab
        tokenizer.add_tokens(new_specials)
        model.resize_token_embeddings(len(tokenizer))  # expand model embeddings

        # 4) “Save” them as attributes for easy access
        tokenizer.start_latent_token   = START
        tokenizer.latent_token         = LAT
        tokenizer.end_latent_token     = END

        tokenizer.start_latent_token_id = tokenizer.convert_tokens_to_ids(START)
        tokenizer.latent_token_id       = tokenizer.convert_tokens_to_ids(LAT)
        tokenizer.end_latent_token_id   = tokenizer.convert_tokens_to_ids(END)

        # mirror them on your model
        model.start_latent_token_id = tokenizer.start_latent_token_id
        model.latent_token_id       = tokenizer.latent_token_id
        model.end_latent_token_id   = tokenizer.end_latent_token_id
        
        # Get the embedding layer correctly
        embedding_layer = model.get_input_embeddings()
        
        # Init the new latent tokens
        vocab = tokenizer.get_vocab()
        # Use torch.no_grad() to safely modify the weights
        with torch.no_grad():
            # copy existing tokens
            embedding_layer.weight[model.start_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
            embedding_layer.weight[model.end_latent_token_id] = embedding_layer.weight[vocab["."]].clone()
    else:
        # tokens already there – still handy to have the ids on the objects
        tokenizer.start_latent_token_id = tokenizer.convert_tokens_to_ids(START)
        tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids(LAT)
        tokenizer.end_latent_token_id = tokenizer.convert_tokens_to_ids(END)
        model.start_latent_token_id = tokenizer.start_latent_token_id
        model.latent_token_id = tokenizer.latent_token_id
        model.end_latent_token_id = tokenizer.end_latent_token_id
        print("Special tokens already present – skipping re-initialisation.")
    
    # Define two prompts for batch processing
    prompts = [
        "User: What is the capital of Germany?",
        "User: Albert is wondering how much pizza he can eat in one day. \
            He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. \
                If he eats it all, how many pieces does he eat that day?"
    ]
    
    # Process both prompts
    prompts = [p + "\nAssistant: " for p in prompts]
    
    # Tokenize batch of prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    # Print input shapes
    print(f"Prompt IDs Shape: {prompt_ids.shape}")

    # Generate responses for the batch
    completion_token_ids, prompt_completion_embeds = model.generate(
        prompt_ids,
        attention_mask=attention_mask,
        num_latent_steps=20,  # Set the number of latent steps        
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        generation_config=None
    )

    # Print the prompt_completion_embeds shape
    print(f"Prompt Completion Embeds Shape: {prompt_completion_embeds.shape}")
    # Print the completion_token_ids shape
    print(f"Completion Token IDs Shape: {completion_token_ids.shape}")
    # Decode the batch of generated responses
    response_texts = tokenizer.batch_decode(completion_token_ids, skip_special_tokens=False)
    
    # Print each prompt and its corresponding response
    for i, (prompt, response) in enumerate(zip(prompts, response_texts)):
        print(f"Prompt {i+1}: {prompt!r}")
        print(f"Response {i+1}: {response!r}")
        print("-" * 50)