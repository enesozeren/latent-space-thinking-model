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
    model = LatentReasoner.from_pretrained(cfg["model"]["base_model_name_or_path"])
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
    batch_size = cfg.get("generation", {}).get("batch_size", 2)  # Default batch size if not specified
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, "value_model_training_data.h5")
    
    # Process training data
    total_examples = len(data['train'])
    print(f"Processing {total_examples} examples with batch size {batch_size}...")
    
    # Initialize counters for valid examples
    valid_examples_count = 0
    
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
        
        # Process examples in batches
        for batch_start in tqdm(range(0, total_examples, batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, total_examples)
            batch_examples = data['train'][batch_start:batch_end]
            
            # Prepare batch data
            batch_prompts = batch_examples["prompt"]
            batch_answers = batch_examples["answer"]
            batch_indices = list(range(batch_start, batch_end))
            
            try:
                # Tokenize batch prompts
                tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
                prompt_ids = tokenized["input_ids"].to(device)
                attention_mask = tokenized["attention_mask"].to(device)
                
                # Generate responses for the batch
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
                
                # Process batch results - handle entire batch at once
                batch_latent_vectors = []
                batch_accuracy_rewards = []
                batch_format_rewards = []
                batch_valid_indices = []
                
                # Get actual prompt lengths for each example in the batch
                prompt_lengths = attention_mask.sum(dim=1).cpu().numpy()
                
                # Decode all responses in the batch at once using batch_decode
                batch_response_texts = tokenizer.batch_decode(completion_token_ids, skip_special_tokens=True)
                
                # Calculate rewards for entire batch at once
                batch_accuracy_rewards_raw = accuracy_reward(batch_response_texts, batch_answers)
                batch_format_rewards_raw = latent_format_reward(batch_response_texts)
                
                # Process each example in the batch
                for i, example_idx in enumerate(batch_indices):
                    try:
                        # Extract latent step vectors
                        prompt_length = prompt_lengths[i]
                        latent_start_idx = prompt_length + 1
                        latent_end_idx = latent_start_idx + num_latent_steps
                        
                        if prompt_completion_embeds.shape[1] < latent_end_idx:
                            continue
                        
                        latent_vectors = prompt_completion_embeds[i, latent_start_idx:latent_end_idx, :].cpu().numpy()
                        
                        # Get rewards from batch calculations
                        accuracy_score = batch_accuracy_rewards_raw[i] if batch_accuracy_rewards_raw[i] is not None else 0.0
                        format_score = batch_format_rewards_raw[i] if batch_format_rewards_raw[i] else 0.0
                        
                        # Collect batch results
                        batch_latent_vectors.append(latent_vectors)
                        batch_accuracy_rewards.append(accuracy_score)
                        batch_format_rewards.append(format_score)
                        batch_valid_indices.append(example_idx)
                        
                    except Exception as e:
                        print(f"Error processing example {example_idx} in batch: {e}")
                        continue
                
                # Save batch results if any valid examples
                if batch_latent_vectors:
                    # Convert to numpy arrays
                    batch_latent_vectors = np.array(batch_latent_vectors)
                    batch_accuracy_rewards = np.array(batch_accuracy_rewards)
                    batch_format_rewards = np.array(batch_format_rewards)
                    batch_valid_indices = np.array(batch_valid_indices)
                    
                    # Resize datasets to accommodate new batch data
                    current_size = latent_vectors_ds.shape[0]
                    new_size = current_size + len(batch_latent_vectors)
                    
                    latent_vectors_ds.resize((new_size, num_latent_steps, latent_dim))
                    accuracy_rewards_ds.resize((new_size,))
                    format_rewards_ds.resize((new_size,))
                    example_ids_ds.resize((new_size,))
                    
                    # Add batch data
                    latent_vectors_ds[current_size:new_size] = batch_latent_vectors
                    accuracy_rewards_ds[current_size:new_size] = batch_accuracy_rewards
                    format_rewards_ds[current_size:new_size] = batch_format_rewards
                    example_ids_ds[current_size:new_size] = batch_valid_indices
                    
                    valid_examples_count += len(batch_latent_vectors)
                    
                    # Flush data to disk periodically
                    if valid_examples_count % 1000 == 0:
                        hf.flush()
                        print(f"Processed {valid_examples_count} valid examples so far...")
                
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                continue
        
        # Store metadata as attributes
        hf.attrs['total_examples'] = total_examples
        hf.attrs['valid_examples'] = valid_examples_count
        hf.attrs['num_latent_steps'] = num_latent_steps
        hf.attrs['latent_dim'] = latent_dim
        hf.attrs['max_length'] = max_length
        hf.attrs['temperature'] = temperature
        hf.attrs['top_p'] = top_p
        hf.attrs['batch_size'] = batch_size
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