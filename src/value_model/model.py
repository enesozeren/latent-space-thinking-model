import json
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from transformers.modeling_outputs import ModelOutput
from src.latent_reasoner.model import LatentReasoner

class ValueHeadModel(nn.Module):
    '''
        A value model which maps the input embedding vector 
        and maps it to a reward value between 0 and 1
    '''
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        if hidden_dims != []:
        # Build hidden layers with dropout
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
        
        # Output layer (maps to single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        logits = self.mlp(x)  # shape: (batch, 1)
        return logits


class ValueModel(nn.Module):
    def __init__(self, latent_reasoner: LatentReasoner, value_head: ValueHeadModel):
        """
        Value model that wraps a LatentReasoner with a pre-trained value head.
        
        Args:
            latent_reasoner: Pre-trained LatentReasoner model
            value_head: Pre-trained ValueHeadModel model
        """
        super().__init__()
        self.latent_reasoner = latent_reasoner
        self.config = latent_reasoner.config
        self.value_head = value_head
    
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Forward pass through the value model.
        
        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            values
        """
        # Get hidden states from the latent reasoner
        # Call the latent reasoner's forward method to get hidden states
        outputs = self.latent_reasoner.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # Get the last layer hidden states
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # Pass through value head to get values for each token
        values = self.value_head(last_hidden_states)  # [batch_size, seq_len, 1]
        values = values.squeeze(-1)  # [batch_size, seq_len]
        
        return values
    
    def save_pretrained(self, save_directory: str):
        """Save the value model."""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the latent reasoner
        self.latent_reasoner.save_pretrained(os.path.join(save_directory, "latent_reasoner"))
        
        # Save the value head
        torch.save(self.value_head.state_dict(), os.path.join(save_directory, "value_head.pt"))
        
        # Save value head config (optional, for reconstruction)
        config_dict = {
            "input_dim": self.config.hidden_size,
            # Note: hidden_dims and dropout info is lost unless stored separately
        }
        with open(os.path.join(save_directory, "value_model_config.json"), "w") as f:
            json.dump(config_dict, f)

    def freeze_value_head(self):
        """Freeze the value head - only LatentReasoner will be trainable"""
        for param in self.value_head.parameters():
            param.requires_grad = False
    
    def freeze_latent_reasoner(self):
        """Freeze the LatentReasoner - only value head will be trainable"""
        for param in self.latent_reasoner.parameters():
            param.requires_grad = False
