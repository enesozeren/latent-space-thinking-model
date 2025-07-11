import yaml
import os
import logging
from datetime import datetime

import torch
import torch.distributed as dist

def load_config(config_path):
    """Load and return the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Setup logging configuration."""
    base_output_dir = config["training"]["output_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, timestamp)
    
    # Update config for output_dir
    config["training"]["output_dir"] = output_dir

    # Ensure the output directory exists before writing the log file
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler()
        ],
        force=True  # override any previous logging configuration
    )

    logging.info("Logging initialised – saving to %s", log_path)
    return output_dir


def _assing_token_ids_to_tokenizer_model(tokenizer, model, special_tokens_dict):
    """Assign token ids to the tokenizer and model."""
    for attr_name, tok_str in special_tokens_dict.items():
        # e.g. attr_name = "start_latent_token", tok_str = "<|start-latent|>"
        tok_id = tokenizer.convert_tokens_to_ids(tok_str)

        # Ensure both the *_token and *_token_id attrs exist on the tokenizer
        setattr(tokenizer, attr_name, tok_str)
        setattr(tokenizer, f"{attr_name}_id", tok_id)

        # On the model we usually only need the *_token_id handle
        setattr(model, f"{attr_name}_id", tok_id)

    logging.info("Token ids assigned to tokenizer and model.")


def setup_latent_tokens(model, tokenizer, is_latent_reasoner: bool = False):
    """Setup special tokens for latent reasoning."""

    if is_latent_reasoner:
        START_LATENT = "<|start-latent|>"
        LATENT = "<|latent|>"
        END_LATENT = "<|end-latent|>"
        latent_specials_dict = {
            "start_latent_token": START_LATENT,
            "latent_token": LATENT,
            "end_latent_token": END_LATENT,
        }
        latent_specials_list = list(latent_specials_dict.values())

        vocab = tokenizer.get_vocab()
        missing_tokens = [tok for tok in latent_specials_list if tok not in vocab]
        
        if missing_tokens:
            # Add tokens as normal tokens to the vocabulary
            for token in missing_tokens:
                tokenizer.add_tokens([token])
            
            # Resize model embeddings to accommodate new tokens
            model.resize_token_embeddings(len(tokenizer))
            logging.info(f"Added {len(missing_tokens)} new normal tokens: {missing_tokens}")
        else:
            logging.info("Latent tokens already present in vocabulary – skipping re-initialisation.")

        _assing_token_ids_to_tokenizer_model(tokenizer, model, latent_specials_dict)

    return model, tokenizer


def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # fall back to environment variable when distributed is not yet initialised
    rank_env = os.environ.get("RANK")
    if rank_env is not None:
        try:
            return int(rank_env) == 0
        except ValueError:
            pass
    return True


def count_max_total_latents(
    start_num_latents, add_latents_delta: int, current_epoch: int, 
    current_batch: int, num_batches_this_epoch: int, max_num_latents: int
    ):
    """Count the maximum total number of latents in the update cycle."""
    # Example case
    # start_num_latents = 0
    # current_epoch = 0
    # num_batches_this_epoch = 30
    # add_latents_delta = 4
    # 30 // 4 = 7
    # 0-6: 1, 7-13: 2, 14-20: 3, 21-27: 4, 28-30: 5
    numb_of_latents_from_last_epoch = current_epoch * add_latents_delta # since epoch starts with 0
    interval_to_add_latent = num_batches_this_epoch // add_latents_delta
    numb_of_latents_to_add = current_batch // interval_to_add_latent + 1
    total_latents = start_num_latents + numb_of_latents_from_last_epoch + numb_of_latents_to_add
    total_latents = min(total_latents, max_num_latents)

    return total_latents