import argparse
import os
from datetime import datetime
import torch
import h5py
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

from src.train.rewards import latent_format_reward, accuracy_reward
from src.data_process.process_data import prepare_dataset_rl
from src.latent_reasoner.model import LatentReasoner
from src.train.utils import load_config, setup_latent_tokens

def create_value_model_training_data(config_path: str) -> None:
    cfg = load_config(config_path)
    # Model and tokenizer
    # Then create the model with this config
    model = LatentReasoner.from_pretrained(cfg["model"]["base_model_name_or_path"])
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"])
    
    # Set up special tokens for think, answer and latent reasoning
    model, tokenizer = setup_latent_tokens(model=model, 
                                            tokenizer=tokenizer, 
                                            is_latent_reasoner=cfg["model"]["is_latent_reasoner"])
    
    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Create a dataloader for question and answers
    data = prepare_dataset_rl(config=cfg, is_latent_reasoner=True)
    
    # Configuration for generation
    num_latent_steps = cfg["model"]["num_latent_steps"]
    temperature = cfg["generation"]["temperature"]
    top_p = cfg["generation"]["top_p"]
    max_length = cfg["generation"]["max_length"]
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, "value_model_training_data.h5")
    
    # Process training data
    total_examples = len(data['train'])
    print(f"Processing {total_examples} examples...")
    
    # Initialize counters for valid examples
    valid_examples_count = 0
    
    # First pass: determine dimensions by processing one example
    first_example = data['train'][0]
    prompt = first_example["prompt"]
    tokenized = tokenizer(prompt, return_tensors="pt", padding=True)
    prompt_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # Get dimensions
    latent_dim = model.get_input_embeddings().embedding_dim
    
    # Create HDF5 file with initial structure
    with h5py.File(output_file, 'w') as hf:
        # Create datasets with initial size 0, but allow unlimited growth
        latent_vectors_ds = hf.create_dataset(
            'latent_vectors', 
            shape=(0, num_latent_steps, latent_dim),
            maxshape=(None, num_latent_steps, latent_dim),
            dtype=np.float32,
            compression='gzip',
            compression_opts=6,
            chunks=True
        )
        
        accuracy_rewards_ds = hf.create_dataset(
            'accuracy_rewards',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            compression='gzip',
            chunks=True
        )
        
        format_rewards_ds = hf.create_dataset(
            'format_rewards',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            compression='gzip',
            chunks=True
        )
        
        example_ids_ds = hf.create_dataset(
            'example_ids',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            compression='gzip',
            chunks=True
        )
        
        # Process all examples and append to datasets
        for idx, example in tqdm(enumerate(data['train']), total=total_examples, desc="Processing examples"):
            
            try:
                prompt = example["prompt"]
                
                # Tokenize prompt
                tokenized = tokenizer(prompt, return_tensors="pt", padding=True)
                prompt_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                
                # Generate responses
                with torch.no_grad():
                    completion_token_ids, prompt_completion_embeds = model.generate(
                        prompt_ids,
                        attention_mask=attention_mask,
                        num_latent_steps=num_latent_steps,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        generation_config=None,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Extract latent step vectors
                prompt_length = prompt_ids.shape[1]
                latent_start_idx = prompt_length + 1
                latent_end_idx = latent_start_idx + num_latent_steps
                
                if prompt_completion_embeds.shape[1] < latent_end_idx:
                    continue
                
                latent_vectors = prompt_completion_embeds[0, latent_start_idx:latent_end_idx, :].cpu().numpy()
                response_text = tokenizer.decode(completion_token_ids[0], skip_special_tokens=False)
                
                # Calculate rewards
                accuracy_rewards = accuracy_reward([response_text], [example["answer"]])
                accuracy_score = accuracy_rewards[0] if accuracy_rewards[0] is not None else 0.0
                
                format_rewards = latent_format_reward([response_text])
                format_score = format_rewards[0] if format_rewards else 0.0
                
                # Resize datasets to accommodate new data
                current_size = latent_vectors_ds.shape[0]
                new_size = current_size + 1
                
                latent_vectors_ds.resize((new_size, num_latent_steps, latent_dim))
                accuracy_rewards_ds.resize((new_size,))
                format_rewards_ds.resize((new_size,))
                example_ids_ds.resize((new_size,))
                
                # Add new data
                latent_vectors_ds[current_size] = latent_vectors
                accuracy_rewards_ds[current_size] = accuracy_score
                format_rewards_ds[current_size] = format_score
                example_ids_ds[current_size] = idx
                
                valid_examples_count += 1
                
                # Flush data to disk periodically to avoid memory buildup
                if valid_examples_count % 1000 == 0:
                    hf.flush()
                
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                continue
        
        # Store metadata as attributes
        hf.attrs['total_examples'] = total_examples
        hf.attrs['valid_examples'] = valid_examples_count
        hf.attrs['num_latent_steps'] = num_latent_steps
        hf.attrs['latent_dim'] = latent_dim
        hf.attrs['max_length'] = max_length
        hf.attrs['temperature'] = temperature
        hf.attrs['top_p'] = top_p
        hf.attrs['created_at'] = datetime.now().isoformat()
        hf.attrs['model_name'] = cfg["model"]["base_model_name_or_path"]
    
    print(f"Completed! Processed {total_examples} examples, saved {valid_examples_count} valid examples.")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(description="Create the dataset for value model training")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/value_model/value_model_training_data_creation.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    create_value_model_training_data(cli_args.config)