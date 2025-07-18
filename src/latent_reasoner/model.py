import types
import torch
import gc
from dataclasses import dataclass
from typing import Optional
from transformers import Qwen2ForCausalLM, Qwen2Config
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
        attention_mask=None,
        track_grad=False
    ):
        """
        1. Append <|start-latent|>
        2. Run the latent-token loop
        3. Append <|end-latent|>
        Returns: inputs_embeds, attention_mask  (same shapes)
        """

        if num_latent_steps == 0:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return inputs_embeds, attention_mask

        batch_size = input_ids.size(0)
        device = input_ids.device
        hidden_size = self.config.hidden_size

        # Pre-allocate the final tensors to avoid repeated concatenations
        final_seq_length = input_ids.size(1) + 1 + num_latent_steps + 1  # +1 for start, +num_latent_steps, +1 for end
        embed_dtype = self.get_input_embeddings().weight.dtype
        inputs_embeds = torch.zeros((batch_size, final_seq_length, hidden_size), 
                                  dtype=embed_dtype, 
                                  device=device)
        mask_dtype = torch.long if attention_mask is None else attention_mask.dtype
        # Attention mask is 1 for all tokens
        attention_mask = torch.ones((batch_size, final_seq_length), 
                                   dtype=mask_dtype, 
                                   device=device)

        # Add the start latent token <|start-latent|>
        start_latents = torch.full((batch_size, 1), self.start_latent_token_id, dtype=input_ids.dtype, device=device)
        input_ids_with_start = torch.cat([input_ids, start_latents], dim=1)
        
        # Get embeddings for the original input + start token
        initial_embeds = self.get_input_embeddings()(input_ids_with_start)
        current_length = initial_embeds.size(1)
        
        # Copy initial embeddings to the pre-allocated tensor
        inputs_embeds[:, :current_length, :] = initial_embeds

        context_manager = torch.enable_grad() if track_grad else torch.no_grad()
        with context_manager:
            # First, process the initial sequence (input + start token) to get initial KV cache
            initial_outputs = super(LatentReasoner, self).forward(
                inputs_embeds=initial_embeds,
                attention_mask=attention_mask[:, :current_length],
                output_hidden_states=True,
                use_cache=True
            )
            
            # Get the initial KV cache
            past_key_values = initial_outputs.past_key_values
            
            # Get the hidden state for the last token (start latent token)
            last_hidden = initial_outputs.hidden_states[-1][:, -1:, :]
            if not track_grad:
                last_hidden = last_hidden.detach()
            
            # Store the hidden state as the first latent embedding
            inputs_embeds[:, current_length:current_length+1, :] = last_hidden
            attention_mask[:, current_length] = 1
            current_length += 1
            
            del initial_outputs
            
            # Now process each latent step using KV cache
            for step_idx in range(num_latent_steps - 1):  # -1 because we already processed the first latent token
                # Only pass the new latent token embedding to the forward pass
                new_latent_embed = inputs_embeds[:, current_length-1:current_length, :]
                new_attention_mask = attention_mask[:, :current_length]
                
                outputs = super(LatentReasoner, self).forward(
                    inputs_embeds=new_latent_embed,
                    attention_mask=new_attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )
                
                # Update past_key_values for next iteration
                past_key_values = outputs.past_key_values
                
                # Get the hidden state for the new latent token
                step_hidden = outputs.hidden_states[-1][:, -1:, :]
                
                # Detach to prevent gradient accumulation across steps
                if not track_grad:
                    step_hidden = step_hidden.detach()
                
                # Add the latent embedding to inputs_embeds at the current position
                inputs_embeds[:, current_length:current_length+1, :] = step_hidden
                
                # Update attention mask for the new latent token
                attention_mask[:, current_length] = 1
                
                # Increment current_length for next iteration
                current_length += 1
                
                # Clear intermediate variables
                del step_hidden
                
                # Clear intermediate variables
                del outputs

        # Add the end latent token <|end-latent|>
        end_latent_embeds = self.get_input_embeddings()(
            torch.full(
                (batch_size, 1),
                self.end_latent_token_id,
                dtype=torch.long,
                device=device,
            )
        )
        inputs_embeds[:, current_length:current_length+1, :] = end_latent_embeds
        attention_mask[:, current_length] = 1

        return inputs_embeds, attention_mask

    def generate(self, input_ids=None, attention_mask=None, num_latent_steps=0, **gen_kwargs):
        """
        Generate text using the model with latent reasoning.
        Returns completion token ids and prompt+completion embeddings.
        """
        # Clear CUDA cache to avoid memory issues
        torch.cuda.empty_cache()

        assert input_ids is not None, "input_ids must be provided for LatentReasoner.generate()"

        if not isinstance(num_latent_steps, int) or num_latent_steps < 0:
            raise ValueError("`num_latent_steps` must be a non-negative integer")
        
        try:
            max_new_tokens = gen_kwargs['generation_config'].max_new_tokens
        except (KeyError, AttributeError):
            # either 'generation_config' isn't in the dict,
            # or it doesn't have a .max_new_tokens attribute
            max_new_tokens = gen_kwargs.get('max_new_tokens', 0)

        with torch.no_grad():
            # augment with latent steps
            inputs_embeds, attention_mask = self._prepare_latent_context(
                num_latent_steps=num_latent_steps,
                input_ids=input_ids,
                attention_mask=attention_mask,
                track_grad=False
            )

        if max_new_tokens and max_new_tokens > 0:
            # Call the base generator
            # The base generater will return only the generated token ids when we pass inputs_embeds
            # since it doesn't have the input token ids but only the input embeddings
            with torch.no_grad():  
                generation_output = super().generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                # Handle different return types from generate
                if hasattr(generation_output, 'sequences'):
                    completion_language_token_ids = generation_output.sequences
                else:
                    completion_language_token_ids = generation_output
                completion_language_token_ids = completion_language_token_ids.detach()
            
            # Append the completion embeddings to the inputs_embeds
            with torch.no_grad():
                completion_embeds = self.get_input_embeddings()(completion_language_token_ids)
            prompt_completion_embeds = torch.cat([inputs_embeds, completion_embeds], dim=1)
            
            # Clean up intermediate tensors
            del completion_embeds
        else:
            prompt_completion_embeds = inputs_embeds
        
        del inputs_embeds
        
        if num_latent_steps > 0:
            # Prepend latent tokens (<|start-latent|>, <|latent|>s, <|end-latent|>)
            batch_size = input_ids.size(0)
            device = input_ids.device
            dtype = input_ids.dtype
            
            # Create tensor of latent token ids [start_latent, latent, latent, ..., end_latent]
            if self.latent_token_id is None:
                raise ValueError("latent_token_id must be set before calling generate")
            latent_ids = torch.ones((batch_size, num_latent_steps+2),  # +2 for start and end latent tokens
                                    dtype=dtype, device=device) * self.latent_token_id
            latent_ids[:, 0] = self.start_latent_token_id
            latent_ids[:, -1] = self.end_latent_token_id
            
            # Concatenate with completion language tokens
            if max_new_tokens and max_new_tokens > 0:
                completion_token_ids = torch.cat([latent_ids, completion_language_token_ids], dim=1)
            else:
                completion_token_ids = latent_ids
        else:
            # No latent reasoning → return only the language tokens
            completion_token_ids = completion_language_token_ids

        # Final cleanup
        torch.cuda.empty_cache()

        return completion_token_ids, prompt_completion_embeds
    
    def sft_forward(
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

        if num_latent_steps > 0:
            # Find the latent token indices
            latent_idxs = (input_ids[0] == self.latent_token_id).nonzero(as_tuple=True)[0]

            # Get the prompt_ids
            prompt_ids = input_ids[0, :latent_idxs[0] - 1] # -1 because of the start latent token and exclusive list indexing after :
            # Get the answer part after the latent tokens
            answer_ids = input_ids[0, latent_idxs[-1] + 2:] # +2 because of the end latent token and inclusive list indexing before :

            # Call the generate method to get the prompt + latent step embeddings
            prompt_completion_embeds, _ = self._prepare_latent_context(
                input_ids=prompt_ids.unsqueeze(0),
                num_latent_steps=num_latent_steps,
                track_grad=False # since latent steps do not contribute to the loss
            )
            
            answer_embeds = self.get_input_embeddings()(answer_ids.unsqueeze(0))

            # Concatenate the prompt + latent + answer embeddings
            inputs_embeds = torch.cat([prompt_completion_embeds, answer_embeds], dim=1)
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Call the parent forward method
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels
        )
        
        return LatentReasonerOutput(loss=outputs.loss, logits=outputs.logits)
