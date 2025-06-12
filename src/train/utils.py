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


def is_rank_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_special_tokens(model, tokenizer, is_latent_reasoner: bool = False):
    """Setup special tokens for think, answer and latent reasoning."""
    START_THINK = "<think>"
    END_THINK = "</think>"
    START_ANSWER = "<answer>"
    END_ANSWER = "</answer>"
    think_answer_specials = [START_THINK, END_THINK, START_ANSWER, END_ANSWER]

    # Add think/answer special tokens if not already present
    if not all(tok in tokenizer.get_vocab() for tok in think_answer_specials):
        # Add them to the tokenizer's vocab
        tokenizer.add_tokens(think_answer_specials)
        model.resize_token_embeddings(len(tokenizer))
        # Save them as attributes for easy access
        tokenizer.start_think_token = START_THINK
        tokenizer.end_think_token = END_THINK
        tokenizer.start_answer_token = START_ANSWER
        tokenizer.end_answer_token = END_ANSWER
        tokenizer.start_think_token_id = tokenizer.convert_tokens_to_ids(START_THINK)
        tokenizer.end_think_token_id = tokenizer.convert_tokens_to_ids(END_THINK)
        tokenizer.start_answer_token_id = tokenizer.convert_tokens_to_ids(START_ANSWER)
        tokenizer.end_answer_token_id = tokenizer.convert_tokens_to_ids(END_ANSWER)
    else:
        # tokens already there – still handy to have the ids on the objects
        tokenizer.start_think_token_id = tokenizer.convert_tokens_to_ids(START_THINK)
        tokenizer.end_think_token_id = tokenizer.convert_tokens_to_ids(END_THINK)
        tokenizer.start_answer_token_id = tokenizer.convert_tokens_to_ids(START_ANSWER)
        tokenizer.end_answer_token_id = tokenizer.convert_tokens_to_ids(END_ANSWER)
        logging.info("Think/answer special tokens already present – skipping re-initialisation.")

    if is_latent_reasoner:
        START_LATENT = "<|start-latent|>"
        LATENT = "<|latent|>"
        END_LATENT = "<|end-latent|>"
        latent_specials = [START_LATENT, LATENT, END_LATENT]
        # Add latent special tokens if not already present
        if not all(tok in tokenizer.get_vocab() for tok in latent_specials):
            # Add them to the tokenizer's vocab
            tokenizer.add_tokens(latent_specials)
            model.resize_token_embeddings(len(tokenizer))  # expand model embeddings

            # Save them as attributes for easy access
            tokenizer.start_latent_token = START_LATENT
            tokenizer.latent_token = LATENT
            tokenizer.end_latent_token = END_LATENT

            tokenizer.start_latent_token_id = tokenizer.convert_tokens_to_ids(START_LATENT)
            tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids(LATENT)
            tokenizer.end_latent_token_id = tokenizer.convert_tokens_to_ids(END_LATENT)

            # mirror them on your model
            model.start_latent_token_id = tokenizer.start_latent_token_id
            model.latent_token_id = tokenizer.latent_token_id
            model.end_latent_token_id = tokenizer.end_latent_token_id
            
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
            tokenizer.start_latent_token_id = tokenizer.convert_tokens_to_ids(START_LATENT)
            tokenizer.latent_token_id = tokenizer.convert_tokens_to_ids(LATENT)
            tokenizer.end_latent_token_id = tokenizer.convert_tokens_to_ids(END_LATENT)
            model.start_latent_token_id = tokenizer.start_latent_token_id
            model.latent_token_id = tokenizer.latent_token_id
            model.end_latent_token_id = tokenizer.end_latent_token_id
            logging.info("Special tokens already present – skipping re-initialisation.")

    return model, tokenizer
