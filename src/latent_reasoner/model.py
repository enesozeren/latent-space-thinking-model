import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, Qwen2Config


class LatentReasoner(Qwen2ForCausalLM):

    def __init__(self, config: Qwen2Config):
        super().__init__(config)

    def _prepare_latent_context(
        self,
        input_ids,
        attention_mask=None,
        num_latent_steps: int = 5,
    ):
        """
        1. Append <|start-latent|>
        2. Run the latent-token loop
        3. Append <|end-latent|>
        Returns: inputs_embeds, attention_mask  (same shapes)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Add the start latent token <|start-latent|>
        start_latents = torch.full((batch_size, 1), self.start_latent_token_id, dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, start_latents], dim=1)

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if attention_mask is None:
            # new length already includes the extra token
            attention_mask = torch.ones_like(input_ids)
        else:
            # caller-provided mask: grow it by one
            start_mask = torch.ones((batch_size, 1),
                                    dtype=attention_mask.dtype,
                                    device=device)
            attention_mask = torch.cat([attention_mask, start_mask], dim=1)
            
        seq_length = inputs_embeds.size(1)
        
        # Handle position_ids
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        for _ in range(num_latent_steps):
            outputs = self(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            step = outputs.hidden_states[-1][:, -1:, :]
            inputs_embeds = torch.cat([inputs_embeds, step], dim=1)
            
            # Update attention mask
            new_mask = torch.ones(inputs_embeds.size(0), 1, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            
            # Update position_ids for the new token position
            new_positions = torch.full((batch_size, 1), position_ids.size(1), dtype=torch.long, device=device)
            position_ids = torch.cat([position_ids, new_positions], dim=1)

        # Add the end latent token <|end-latent|>
        end_latent_embed = self.get_input_embeddings()(
            torch.full(
                (batch_size, 1),
                self.end_latent_token_id,
                dtype=torch.long,
                device=device,
            )
        )
        inputs_embeds = torch.cat([inputs_embeds, end_latent_embed], dim=1)
        new_mask = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        return inputs_embeds, attention_mask

    def generate(self, input_ids=None, attention_mask=None, num_latent_steps: int = 5, **gen_kwargs):
        """
        Generate text using the model with latent reasoning.
        Returns [completion] token ids and [prompt, completion] embeddings.
        """
        if input_ids is not None:
            # augment with latent steps
            inputs_embeds, attention_mask = self._prepare_latent_context(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_latent_steps=num_latent_steps
            )

            # call the base generator
            # The base generater will return only the generated token ids 
            # since it doesn't have the input token ids but only the input embeddings
            completion_token_ids = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # Append the completion embeddings to the inputs_embeds
            completion_embeds = self.get_input_embeddings()(completion_token_ids)
            prompt_completion_embeds = torch.cat([inputs_embeds, completion_embeds], dim=1)

            return completion_token_ids, prompt_completion_embeds
        else:
            raise ValueError("input_ids must be provided for LatentReasoner.generate()")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Then create the model
    model = LatentReasoner.from_pretrained("Qwen/Qwen2.5-0.5B")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    # # Set the pad token to be the same as the eos token since GPT-2 does not have a pad token
    # tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for latent reasoning
    tokenizer.add_special_tokens({
            "additional_special_tokens": [
                "<|start-latent|>",
                "<|end-latent|>",
                "<|latent|>"
            ]
        })
    model.resize_token_embeddings(len(tokenizer))
    
    # Get the embedding layer correctly
    embedding_layer = model.get_input_embeddings()
    
    # Init the new latent tokens
    vocab = tokenizer.get_vocab()
    sid = vocab.get("<|start-latent|>")
    model.start_latent_token_id = sid
    eid = vocab.get("<|end-latent|>")
    model.end_latent_token_id = eid
    lid = vocab.get("<|latent|>")
    model.latent_token_id = lid
    
    # Use torch.no_grad() to safely modify the weights
    with torch.no_grad():
        # copy existing tokens
        embedding_layer.weight[sid] = embedding_layer.weight[vocab["="]].clone()
        embedding_layer.weight[eid] = embedding_layer.weight[vocab[">"]].clone()

    # The input embeddings and lm heads are tied in GPT2. So we don't need to modify the lm head

    # Define two prompts for batch processing
    prompts = [
        "User: What is the capital of Germany?",
        "User: Can you explain quantum computing in simple terms?"
    ]
    
    # Process both prompts
    prompts = [p + "\nAssistant: " for p in prompts]
    
    # Tokenize batch of prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Generate responses for the batch
    completion_token_ids, prompt_completion_embeds = model.generate(
        prompt_ids,
        attention_mask=attention_mask,
        num_latent_steps=5,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.7,
        generation_config=None
    )

    # Print the prompt_completion_embeds shape
    print(f"Prompt Completion Embeds Shape: {prompt_completion_embeds.shape}")
    # Decode the batch of generated responses
    response_texts = tokenizer.batch_decode(completion_token_ids, skip_special_tokens=False)
    
    # Print each prompt and its corresponding response
    for i, (prompt, response) in enumerate(zip(prompts, response_texts)):
        print(f"Prompt {i+1}: {prompt!r}")
        print(f"Response {i+1}: {response!r}")
        print("-" * 50)
