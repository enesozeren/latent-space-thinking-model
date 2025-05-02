"""
Evaluate models on openai/gsm8k and HuggingFaceH4/MATH-500

This script can evaluate:
1. Local model checkpoints from huggingface trl GRPO trainer
2. Models directly from huggingface hub
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

import torch
import random
import numpy as np
from transformers import set_seed
import wandb
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from prompts.prompts import (
    SYSTEM_PROMPT_GSM8K_4_SHOT_EVAL, 
    SYSTEM_PROMPT_MATH500_1_SHOT_EVAL
)
from src.data_process.process_data_eval import prepare_dataset

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Disable flash SDP and enable math SDP for better performance
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on mathematical reasoning benchmarks")
    
    # Model args
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to a local model checkpoint or a model ID from Huggingface Hub",
    )
    
    # Dataset args
    parser.add_argument(
        "--dataset",
        type=str,
        default="openai/gsm8k",
        choices=["openai/gsm8k", "HuggingFaceH4/MATH-500"],
        help="Dataset to evaluate on: 'gsm8k' for openai/gsm8k, 'math500' for HuggingFaceH4/MATH-500",
    )
    
    # Generation args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum token length for generated responses",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    
    # Evaluation args
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Limit evaluation to this many examples (default: use all examples)",
    )
    
    # Output args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation results",
    )
    
    # W&B integration args
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="latent_reasoner_eval",
        help="W&B project name for logging results",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (default: model_name_dataset_eval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    
    return parser.parse_args()

def format_prompt(question: str, dataset_name: str) -> Union[str, List[Dict[str, str]]]:
    """Format the prompt for the model based on the dataset."""
    if dataset_name == "openai/gsm8k":
        system_prompt_for_eval = SYSTEM_PROMPT_GSM8K_4_SHOT_EVAL
    elif dataset_name == "HuggingFaceH4/MATH-500":
        system_prompt_for_eval = SYSTEM_PROMPT_MATH500_1_SHOT_EVAL
    
    return system_prompt_for_eval + "\nUser:" + question + "\nAssistant:"

def load_model_and_tokenizer(args) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer."""
    logger.info(f"Loading model from {args.model_name_or_path}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    
    return model, tokenizer

def generate_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[Union[str, List[Dict[str, str]]]],
    batch_size: int,
    max_length: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """Generate responses in batches."""
    responses = []
    
    # Move model to evaluation mode
    model.eval()

    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i+batch_size]
        
        # For base models, use regular tokenizer
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device=model.device)
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=temperature > 0.0,
            )
        
        # Decode outputs
        batch_responses = []
        for j, output in enumerate(outputs):
            # Get the length of input tokens for this example
            input_length = inputs["input_ids"][j].size(0)
            
            # Get only the generated tokens (excluding input)
            response_tokens = output[input_length:]
            
            # Decode only the response part
            assistant_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            batch_responses.append(assistant_response)
        
        responses.extend(batch_responses)
    
    return responses

def extract_answer_from_response(response: str) -> str:
    """Extract final answer from model's response according to the expected format.
    
    Expected format follows:
    <think>reasoning process</think>
    <answer>\\boxed{final_answer}</answer>
    
    Also handles other formats like direct boxed answers, numerical answers after ####,
    and other common patterns.
    """
    # First try to extract content between <answer> tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_match = answer_pattern.search(response)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
        # Look for \boxed{...} in the answer content
        boxed_match = extract_boxed_content(answer_content)
        if boxed_match:
            return boxed_match
        
        # If there's answer content but no boxed format, return cleaned content
        return clean_answer_text(answer_content)
    
    # Look for any \boxed{...} in the full response if not found in answer tags
    boxed_match = extract_boxed_content(response)
    if boxed_match:
        return boxed_match
    
    # Fallback methods if the expected format is not found
    if "####" in response:
        answer = response.split("####")[-1].strip()
        return clean_answer_text(answer)
    
    # Look for other common answer patterns
    for pattern in ["final answer:", "the answer is", "answer:"]:
        if pattern in response.lower():
            answer_text = response.lower().split(pattern)[-1].strip()
            return clean_answer_text(answer_text)
            
    # Try to extract numerical answer from the last line
    # Often models will put the final numeric answer on the last line
    last_line = response.strip().split('\n')[-1].strip()
    if re.search(r'^\s*\d+\s*$', last_line):  # If last line is just a number
        return last_line.strip()
    
    # Find all numbers in the text and return the last one
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        return numbers[-1]
    
    # If all else fails, return the last line
    return clean_answer_text(last_line)

def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} LaTeX notation."""
    # Standard LaTeX boxed pattern
    boxed_patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",  # Standard LaTeX \boxed{...}
        r"\(\s*\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\)",  # With math delimiters (\boxed{...})
        r"\$\s*\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\$",  # With dollar math delimiters $\boxed{...}$
    ]
    
    for pattern in boxed_patterns:
        boxed_match = re.search(pattern, text, re.DOTALL)
        if boxed_match:
            content = boxed_match.group(1).strip()
            return clean_answer_text(content)
    
    return None

def clean_answer_text(text: str) -> str:
    """Clean up extracted answer text to get the final numeric or text answer."""
    # Remove LaTeX formatting
    text = text.replace('\\text', '').replace('$', '')
    
    # Check if it's just a number (with possible whitespace)
    if re.match(r'^\s*-?\d+\.?\d*\s*$', text):
        return text.strip()
    
    # Try to find the first number in the text
    number_match = re.search(r'-?\d+\.?\d*', text)
    if number_match:
        return number_match.group(0)
    
    # Remove non-essential characters for numerical answers
    cleaned = re.sub(r'[^\d.-]', '', text)
    if cleaned and re.match(r'-?\d+\.?\d*', cleaned):
        return cleaned
    
    # If it's not clearly a number, return the cleaned text
    # Remove common explanation words and punctuation
    cleaned = re.sub(r"(answer|is|equals|=|:|\.|\,)", "", text.lower())
    return cleaned.strip()

def evaluate_responses(predicted_answers: List[str], ground_truth: List[str], full_responses: List[str], tokenizer=None) -> Dict[str, float]:
    """Evaluate model predictions against ground truth answers."""
    correct = 0
    total = len(ground_truth)
    
    # Format compliance metrics
    has_think_tag = 0
    has_answer_tag = 0
    has_boxed = 0
    fully_formatted = 0
    
    # Calculate average response token length
    total_token_length = 0
    
    for pred, gt, full_resp in zip(predicted_answers, ground_truth, full_responses):
        # Check answer correctness
        try:
            pred_val = float(pred)
            gt_val = float(gt)
            if pred_val == gt_val:
                correct += 1
        except ValueError:
            # If conversion fails, do string comparison
            if pred.strip() == gt.strip():
                correct += 1
        
        # Check format compliance
        has_think = bool(re.search(r'<think>.*?</think>', full_resp, re.DOTALL))
        has_answer = bool(re.search(r'<answer>.*?</answer>', full_resp, re.DOTALL))
        has_box = '\\boxed{' in full_resp
        
        if has_think:
            has_think_tag += 1
        if has_answer:
            has_answer_tag += 1
        if has_box:
            has_boxed += 1
        if has_think and has_answer and has_box:
            fully_formatted += 1
            
        # Add response token length if tokenizer is provided
        if tokenizer is not None:
            tokens = tokenizer.encode(full_resp)
            total_token_length += len(tokens)
    
    accuracy = correct / total if total > 0 else 0
    avg_token_length = total_token_length / total if total > 0 and tokenizer is not None else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_token_length": avg_token_length,
        "format_metrics": {
            "has_think_tag_pct": has_think_tag / total if total > 0 else 0,
            "has_answer_tag_pct": has_answer_tag / total if total > 0 else 0,
            "has_boxed_pct": has_boxed / total if total > 0 else 0,
            "fully_formatted_pct": fully_formatted / total if total > 0 else 0,
            "has_think_tag": has_think_tag,
            "has_answer_tag": has_answer_tag,
            "has_boxed": has_boxed,
            "fully_formatted": fully_formatted,
        }
    }

def save_results(
    args,
    dataset,
    responses,
    extracted_answers,
    metrics,
):
    """Save evaluation results to output directory."""
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_name_or_path).name
    # Include dataset name in output folder
    output_dir = Path(args.output_dir) / f"{model_name}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Create examples list for both local save and wandb
    examples_list = []
    examples_path = output_dir / "examples.jsonl"
    with open(examples_path, "w") as f:
        for i, (question, gt_answer, response, extracted_answer) in enumerate(
            zip(dataset["questions"], dataset["answers"], responses, extracted_answers)
        ):
            example = {
                "id": i,
                "question": question,
                "ground_truth": gt_answer,
                "model_response": response,
                "extracted_answer": extracted_answer,
                "correct": extracted_answer.strip() == gt_answer.strip() 
                           or (safe_float_convert(extracted_answer) == safe_float_convert(gt_answer)),
            }
            examples_list.append(example)
            f.write(json.dumps(example) + "\n")
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    # Save config as YAML
    config_yaml_path = output_dir / "config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.safe_dump(vars(args), f)

    # Log to W&B if not disabled
    if not args.no_wandb:
        # Initialize wandb if not already running
        run_name = args.wandb_run_name or f"{model_name}_{args.dataset}_eval"
        run = wandb.init(
            project=args.wandb_project, 
            name=run_name, 
            config=vars(args),
            job_type="evaluation", 
            id=timestamp
        )
        
        # Log metrics to wandb
        wandb.log(metrics)
        
        # Create a wandb Table for examples
        columns = ["id", "question", "ground_truth", "model_response", "extracted_answer", "correct"]
        example_table = wandb.Table(columns=columns)
        
        # Add each example to the table
        for ex in examples_list:
            example_table.add_data(ex["id"], ex["question"], ex["ground_truth"], 
                                  ex["model_response"], ex["extracted_answer"], ex["correct"])
        
        # Log table
        wandb.log({"examples": example_table})
        
        # Upload JSON files to wandb
        wandb.save(str(metrics_path))
        wandb.save(str(examples_path))
        wandb.save(str(config_path))
        # Upload YAML config to wandb
        wandb.save(str(config_yaml_path))
        
        logger.info(f"Results saved to {output_dir} and logged to W&B")
    else:
        logger.info(f"Results saved to {output_dir} (W&B logging disabled)")
        
    return output_dir

def safe_float_convert(value):
    """Safely convert a value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load appropriate dataset based on user selection
    if args.dataset in ("openai/gsm8k", "HuggingFaceH4/MATH-500"):
        dataset = prepare_dataset(args.dataset, args.split, args.num_examples)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    logger.info(f"Evaluating on {args.dataset} dataset")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Format prompts
    prompts = [format_prompt(q, args.dataset) for q in dataset["questions"]]
    
    # Generate responses
    start_time = time.time()
    responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    generation_time = time.time() - start_time
    
    # Extract answers from responses
    extracted_answers = [extract_answer_from_response(resp) for resp in responses]
    
    # Evaluate responses - now passing tokenizer for token length calculation
    metrics = evaluate_responses(extracted_answers, dataset["answers"], responses, tokenizer)
    metrics["generation_time"] = generation_time
    metrics["examples_per_second"] = len(dataset["questions"]) / generation_time
    
    # Log metrics
    logger.info(f"Evaluation metrics on {args.dataset}:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # Log format compliance metrics
    format_metrics = metrics["format_metrics"]
    logger.info(f"Format compliance metrics:")
    logger.info(f"  Has <think> tags: {format_metrics['has_think_tag_pct']:.2%} ({format_metrics['has_think_tag']}/{metrics['total']})")
    logger.info(f"  Has <answer> tags: {format_metrics['has_answer_tag_pct']:.2%} ({format_metrics['has_answer_tag']}/{metrics['total']})")
    logger.info(f"  Has \\boxed{{}} syntax: {format_metrics['has_boxed_pct']:.2%} ({format_metrics['has_boxed']}/{metrics['total']})")
    logger.info(f"  Fully formatted: {format_metrics['fully_formatted_pct']:.2%} ({format_metrics['fully_formatted']}/{metrics['total']})")
    
    # Log token length metric
    logger.info(f"  Average response token length: {metrics.get('avg_token_length', 0):.2f} tokens")
    
    logger.info(f"  Generation time: {generation_time:.2f}s")
    logger.info(f"  Examples per second: {metrics['examples_per_second']:.2f}")
    
    # Save results
    output_dir = save_results(
        args,
        dataset,
        responses,
        extracted_answers,
        metrics,
    )
    
    return metrics

if __name__ == "__main__":
    main()