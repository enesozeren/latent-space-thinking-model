import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, Qwen2Config

class LatentReasonerConfig(Qwen2Config):
    model_type = "latent-reasoner"

    def __init__(self, num_latent_steps=5, **kwargs):
        super().__init__(**kwargs)
        self.num_latent_steps = num_latent_steps

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.num_latent_steps = kwargs.get(
            "num_latent_steps",
            getattr(config, "num_latent_steps", 5)
        )
        return config


class LatentReasoner(Qwen2ForCausalLM):
    supports_inputs_embeds = True
    config_class = LatentReasonerConfig

    def __init__(self, config: LatentReasonerConfig):
        super().__init__(config)
        self.num_latent_steps = config.num_latent_steps

    def _prepare_latent_context(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
    ):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.size()[:2], device=inputs_embeds.device)
            
        batch_size, seq_length = inputs_embeds.size()[:2]
        
        # Handle position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        for _ in range(self.num_latent_steps):
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
            new_mask = torch.ones(inputs_embeds.size(0), 1, device=inputs_embeds.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)
            
            # Update position_ids for the new token position
            new_positions = torch.full((batch_size, 1), position_ids.size(1), dtype=torch.long, device=position_ids.device)
            position_ids = torch.cat([position_ids, new_positions], dim=1)

        return inputs_embeds, attention_mask

    def generate(self, input_ids=None, attention_mask=None, **gen_kwargs):
        if input_ids is not None:
            # 1 augment with latent steps
            inputs_embeds, attention_mask = self._prepare_latent_context(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # 2 make input_ids the SAME length
            pad_id = (
                self.config.pad_token_id
                if self.config.pad_token_id is not None
                else self.config.eos_token_id
            )
            full_input_ids = torch.full(
                (input_ids.size(0), inputs_embeds.size(1)),
                pad_id,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            full_input_ids[:, : input_ids.size(1)] = input_ids

            # 3 call the base generator
            return super().generate(
                input_ids=full_input_ids,        # <- now 244 tokens
                inputs_embeds=inputs_embeds,     # <- also 244 tokens
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # fallback
        return super().generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs,
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a config first with our custom parameters
    config = LatentReasonerConfig.from_pretrained("Qwen/Qwen2.5-1.5B", num_latent_steps=5)
    # Then create the model with this config
    model = LatentReasoner.from_pretrained("Qwen/Qwen2.5-1.5B", config=config)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    
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
    eid = vocab.get("<|end-latent|>")
    lid = vocab.get("<|latent|>")
    
    # Use torch.no_grad() to safely modify the weights
    with torch.no_grad():
        # copy from similar tokens
        if "<|im_start|>" in vocab and sid is not None:
            embedding_layer.weight[sid] = embedding_layer.weight[vocab["<|im_start|>"]].clone()
        if "<|im_end|>" in vocab and eid is not None:
            embedding_layer.weight[eid] = embedding_layer.weight[vocab["<|im_end|>"]].clone()
        # init latent token with 0s since it won't be feed to the model ever
        if lid is not None:
            embedding_layer.weight[lid] = torch.zeros_like(embedding_layer.weight[0])    

    # Define two prompts for batch processing
    prompts = [
        "User: What is the capital of Germany?",
        "User: Can you explain quantum computing in simple terms?"
    ]
    
    # Process both prompts
    prompts = [p + "\nAssistant: " + "<|start-latent|>" for p in prompts]
    
    # Tokenize batch of prompts
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Generate responses for the batch
    responses = model.generate(
        prompt_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
        do_sample=True,
        generation_config=None
    )

    # Decode the batch of generated responses
    response_texts = tokenizer.batch_decode(responses, skip_special_tokens=False)
    
    # Print each prompt and its corresponding response
    for i, (prompt, response) in enumerate(zip(prompts, response_texts)):
        print(f"Prompt {i+1}: {prompt!r}")
        print(f"Response {i+1}: {response!r}")
        print("-" * 50)
