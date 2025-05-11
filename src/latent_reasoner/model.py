# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# class LatentReasoner(AutoModelForCausalLM):
#     def __init__(self, model_name: str, num_latent_steps: int = 5):
#         # Supported model check
#         if model_name not in ["Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]:
#             raise ValueError(f"Model {model_name} is not supported.")

#         self.num_latent_steps = num_latent_steps
#         # Load model and tokenizer
#         self.model_name = model_name
#         self.model = AutoModelForCausalLM.from_pretrained(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#         # Register special tokens: latent start/end + display-only latent marker
#         self.tokenizer.add_special_tokens({
#             "additional_special_tokens": [
#                 "<|start-latent|>",
#                 "<|end-latent|>",
#                 "<|latent|>"
#             ]
#         })
#         # Resize the model's token embeddings to accommodate new tokens
#         self.model.resize_token_embeddings(len(self.tokenizer))
#         # Get the embedding layer
#         self.embeding_layer = self.model.get_input_embeddings()
#         # Init embeddings for new latent tokens
#         self._init_latent_tokens()

#     def _init_latent_tokens(self):
#         with torch.no_grad():
#             vocab = self.tokenizer.get_vocab()
#             sid = vocab.get("<|start-latent|>")
#             eid = vocab.get("<|end-latent|>")
#             lid = vocab.get("<|latent|>")
#             # copy from similar tokens
#             if "<|im_start|>" in vocab and sid is not None:
#                 self.embeding_layer.weight[sid] = self.embeding_layer.weight[vocab["<|im_start|>"]]
#             if "<|im_end|>" in vocab and eid is not None:
#                 self.embeding_layer.weight[eid] = self.embeding_layer.weight[vocab["<|im_end|>"]]
#             # init latent token with 0s since it won't be feed to the model ever
#             if lid is not None:
#                 self.embeding_layer.weight[lid] = torch.zeros_like(self.embeding_layer.weight[0])
    
#     def train(self):
#         self.model.train()

#     def eval(self):
#         self.model.eval()
        
#     def forward(self, input_embeddings, attention_mask):
        
#         output = self.model(
#             inputs_embeds=input_embeddings,
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             return_dict=True
#         )

#         # Get the last hidden state and logits
#         last_hidden_states = output.hidden_states[-1]
#         logits = output.logits

#         return last_hidden_states, logits

#     def generate(
#         self,
#         inputs,
#         attention_mask,
#         output_embedding: bool = False,
#         max_new_tokens: int = None,
#         max_tokens: int = 20,
#         do_sample: bool = False,
#         temperature: float = 1.0,
#         top_k: int = 50,
#         top_p: float = 1.0,
#         min_p: float = 0.0,
#         repetition_penalty: float = 1.0,
#         generation_config = None,
#         prompts: list = None # Currently not supporting VLLM (we dont use the prompts input)
#     ):
#         """
#         Generate method for the LatentReasoner model.
#         First generates latent space tokens, then uses the model's generate method for language space generation.
#         """
#         device = inputs.device
#         batch_size = inputs.size(0)

#         # Deal both max_tokens and max_new_tokens
#         max_new_tokens = max_new_tokens if max_new_tokens is not None else max_tokens

#         # Special token ids
#         # start_latent_id = self.tokenizer.get_vocab()["<|start-latent|>"]
#         end_latent_id = self.tokenizer.get_vocab()["<|end-latent|>"]
#         # latent_id = self.tokenizer.get_vocab()["<|latent|>"]
#         # padding_id = self.tokenizer.pad_token_id

#         # Get input embeddings
#         inputs_embeds = self.embeding_layer(inputs)

#         # === Latent space generation ===
#         for _ in range(self.num_latent_steps):
#             last_hidden_states, _ = self.forward(inputs_embeds, attention_mask)
#             # Append last latent embedding
#             inputs_embeds = torch.cat([inputs_embeds, last_hidden_states[:, -1:, :]], dim=1)
#             attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)
        
#         # Append end-of-latent marker
#         end_latent_emb = self.embeding_layer(torch.full((batch_size, 1), end_latent_id, device=device, dtype=torch.long))
#         inputs_embeds = torch.cat([inputs_embeds, end_latent_emb], dim=1)
#         attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=1)

#         # === Language space generation ===
#         # now hand off to HuggingFace's generate()
#         generated_ids = self.model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=attention_mask,
#             max_new_tokens=max_new_tokens,
#             do_sample=do_sample,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p,
#             eos_token_id=self.tokenizer.eos_token_id,
#             pad_token_id=self.tokenizer.pad_token_id,
#         )

#         if output_embedding:
#             # If output_embedding is True, return the generated embeddings including the latent space
#             latent_embeddings = inputs_embeds[:, -self.num_latent_steps:, :]
#             generated_embeddings = torch.cat([latent_embeddings, self.embeding_layer(generated_ids)], dim=1)
#             return generated_embeddings
#         else:
#             return generated_ids

import torch
from transformers import Qwen2ForCausalLM, AutoTokenizer, Qwen2Config

class LatentReasonerConfig(Qwen2Config):
    model_type = "latent-reasoner"
    
    def __init__(self, num_latent_steps=5, **kwargs):
        self.num_latent_steps = num_latent_steps
        super().__init__(**kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # First load the original config
        config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # Make sure to preserve our custom attributes if they're in kwargs
        config.num_latent_steps = kwargs.get("num_latent_steps", 5)
        return config

class LatentReasoner(Qwen2ForCausalLM):
    config_class = LatentReasonerConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_latent_steps = getattr(config, "num_latent_steps", 5)

    def generate(self, input_ids=None, attention_mask=None, **gen_kwargs):
        # 1) Embed the inputs
        embed = self.get_input_embeddings()
        inputs_embeds = embed(input_ids)           # [B, L, D]
        mask = attention_mask                     # [B, L]

        # 2) Do N latent forward-only steps
        for _ in range(self.num_latent_steps):
            # a) forward pass to get hidden states
            out = self(
                inputs_embeds=inputs_embeds,
                attention_mask=mask,
                output_hidden_states=True,
                return_dict=True
            )
            # b) grab the last hidden vector
            step = out.hidden_states[-1][:, -1:, :]   # [B, 1, D]
            # c) append to our “context”
            inputs_embeds = torch.cat([inputs_embeds, step], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)

        # 3) Now call the normal generate on the enriched embeddings
        return super().generate(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            **gen_kwargs
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
