import types
import torch
from dataclasses import dataclass
from typing import Optional
from transformers import Qwen2ForCausalLM, AutoTokenizer, Qwen2Config
from transformers.modeling_outputs import ModelOutput

@dataclass
class LatentReasonerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None

class LatentReasoner(Qwen2ForCausalLM):

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        # Latent token ids to be set later
        self.start_latent_token_id = None
        self.latent_token_id = None
        self.end_latent_token_id = None

    def _prepare_latent_context(
        self,
        num_latent_steps,
        input_ids,
        attention_mask=None
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
            outputs = super(LatentReasoner, self).forward(
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
        end_latent_embeds = self.get_input_embeddings()(
            torch.full(
                (batch_size, 1),
                self.end_latent_token_id,
                dtype=torch.long,
                device=device,
            )
        )
        inputs_embeds = torch.cat([inputs_embeds, end_latent_embeds], dim=1)
        new_mask = torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        return inputs_embeds, attention_mask

    def generate(self, input_ids=None, attention_mask=None, num_latent_steps=0, **gen_kwargs):
        """
        Generate text using the model with latent reasoning.
        Returns completion token ids and prompt+completion embeddings.
        """
        assert input_ids is not None, "input_ids must be provided for LatentReasoner.generate()"

        if not isinstance(num_latent_steps, int) or num_latent_steps < 0:
            raise ValueError("`num_latent_steps` must be a non-negative integer")
        
        try:
            max_new_tokens = gen_kwargs['generation_config'].max_new_tokens
        except (KeyError, AttributeError):
            # either 'generation_config' isn't in the dict,
            # or it doesn't have a .max_new_tokens attribute
            max_new_tokens = gen_kwargs.get('max_new_tokens')

        # augment with latent steps
        inputs_embeds, attention_mask = self._prepare_latent_context(
            num_latent_steps=num_latent_steps,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        if max_new_tokens > 0:
            # call the base generator

            # But the super().generate must use original forward method from Qwen2ForCausalLM
            # save a reference to LatentReasoner forward override
            latent_reasoner_forward = self.forward
            # bind the parent’s forward onto this instance
            base_forward = types.MethodType(Qwen2ForCausalLM.forward, self)
            self.forward = base_forward
            
            # The base generater will return only the generated token ids when we pass inputs_embeds
            # since it doesn't have the input token ids but only the input embeddings
            completion_language_token_ids = super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

            # restore LatentReasoner forward override
            self.forward = latent_reasoner_forward
            
            # Append the completion embeddings to the inputs_embeds
            completion_embeds = self.get_input_embeddings()(completion_language_token_ids)
            prompt_completion_embeds = torch.cat([inputs_embeds, completion_embeds], dim=1)
        else:
            prompt_completion_embeds = inputs_embeds
        
        # Prepend latent tokens (<|start-latent|>, <|latent|>s, <|end-latent|>)
        batch_size = input_ids.size(0)
        device = input_ids.device
        dtype = input_ids.dtype
        
        # Create tensor of latent token ids [start_latent, latent, latent, ..., end_latent]
        latent_ids = torch.ones((batch_size, num_latent_steps+2),  # +2 for start and end latent tokens
                                dtype=dtype, device=device) * self.latent_token_id
        latent_ids[:, 0] = self.start_latent_token_id
        latent_ids[:, -1] = self.end_latent_token_id
        
        # Concatenate with completion language tokens
        if max_new_tokens > 0:
            completion_token_ids = torch.cat([latent_ids, completion_language_token_ids], dim=1)
        else:
            completion_token_ids = latent_ids

        return completion_token_ids, prompt_completion_embeds
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        **kwargs
    ):
        """
        Override forward to handle latent reasoning.
        """
        # Only batch size 1 is supported for now
        if input_ids is not None and input_ids.size(0) != 1:
            raise ValueError("LatentReasoner only supports batch size of 1 for now.")
        if inputs_embeds is not None:
            raise ValueError("LatentReasoner's forward method does not support inputs_embeds yet. " \
            "Please use base model's foorward method if you need it.")
        
        # Calculate the number of latent steps by counting the latent tokens
        num_latent_steps = (input_ids[0] == self.latent_token_id).sum().item()

        # Find the start latent token index
        start_latent_index = (input_ids[0] == self.start_latent_token_id).nonzero(as_tuple=True)[0]
        end_latent_index = (input_ids[0] == self.end_latent_token_id).nonzero(as_tuple=True)[0]

        # Get the prompt_ids
        prompt_ids = input_ids[0, :start_latent_index]
        # Get the answer part after the latent tokens
        answer_ids = input_ids[0, end_latent_index + 1:]

        # Call the generate method to get the prompt + latent step embeddings
        completion_token_ids, prompt_completion_embeds = self.generate(
            input_ids=prompt_ids.unsqueeze(0),
            num_latent_steps=num_latent_steps,
            max_new_tokens=0,  # No new tokens, just latent steps
        )
        
        answer_embeds = self.get_input_embeddings()(answer_ids.unsqueeze(0))

        # Concatenate the prompt + latent + answer embeddings
        inputs_embeds = torch.cat([prompt_completion_embeds, answer_embeds], dim=1)
        # Call the parent forward method
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels
        )

        # Return the outputs with the loss and logits
        return LatentReasonerOutput(
            loss=outputs.loss,
            logits=outputs.logits
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Then create the model
    model = LatentReasoner.from_pretrained("Qwen/Qwen2.5-0.5B")
    # Set the number of latent steps for the model
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    # Define new tokens for latent steps
    START = "<|start-latent|>"
    LAT   = "<|latent|>"
    END   = "<|end-latent|>"
    new_specials = [START, LAT, END]

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
        embedding_layer.weight[model.start_latent_token_id] = embedding_layer.weight[vocab["="]].clone()
        embedding_layer.weight[model.end_latent_token_id] = embedding_layer.weight[vocab[">"]].clone()

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
    # Print input shapes
    print(f"Prompt IDs Shape: {prompt_ids.shape}")

    # Generate responses for the batch
    completion_token_ids, prompt_completion_embeds = model.generate(
        prompt_ids,
        attention_mask=attention_mask,
        num_latent_steps=0,  # Set the number of latent steps        
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
