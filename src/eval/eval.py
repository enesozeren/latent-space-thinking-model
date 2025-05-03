"""
Evaluate models on openai/gsm8k and HuggingFaceH4/MATH-500

This script can evaluate:
1. Local model checkpoints from huggingface trl GRPO trainer
2. Models directly from huggingface hub
"""

import argparse
import re
import sys
import time

from typing import List, Dict, Any, Union, Optional, Tuple

import torch
import random
import numpy as np
from transformers import set_seed
import yaml
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from prompts.prompts import (
    SYSTEM_PROMPT_GSM8K_4_SHOT_EVAL, 
    SYSTEM_PROMPT_MATH500_1_SHOT_EVAL
)
from src.data_process.process_data_eval import prepare_dataset
from src.eval.eval_utils import save_results

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on mathematical reasoning benchmarks")
    
    # Model args
    parser.add_argument(
        "--model_name_or_path", type=str, required=True,
        help="Path to a local model checkpoint or a model ID from Huggingface Hub",
    )
    
    # Dataset args
    parser.add_argument(
        "--dataset", type=str, default="openai/gsm8k", 
        choices=["openai/gsm8k", "HuggingFaceH4/MATH-500"],
        help="Dataset to evaluate on: 'gsm8k' for openai/gsm8k, 'math500' for HuggingFaceH4/MATH-500",
    )
    
    # Generation args
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Maximum token length for generated responses",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for evaluation",
    )
    
    # Evaluation args
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "train"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--num_examples", type=int, default=None,
        help="Limit evaluation to this many examples (default: use all examples)",
    )
    
    # Output args
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory to save evaluation results",
    )
    
    # W&B integration args
    parser.add_argument(
        "--wandb_project", type=str, default="latent_reasoner_eval",
        help="W&B project name for logging results",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="W&B run name (default: model_name_dataset_eval)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
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
    """
    # First try to extract content between <answer> tags
    answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    answer_match = answer_pattern.search(response)
    boxed_match = ""

    if answer_match:
        answer_content = answer_match.group(1).strip()
        # Look for \boxed{...} in the answer content
        boxed_match = extract_boxed_content(answer_content)
    
    return boxed_match

def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}"""

    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    m_box = re.search(boxed_pattern, text, re.DOTALL)

    if m_box:
        boxed_content = m_box.group(1).strip()
        return boxed_content
    else:
        # No boxed expression found → return empty string
        return ""

def evaluate_responses(predicted_answers: List[str], ground_truth: List[str], full_responses: List[str], tokenizer=None) -> Dict[str, Any]:
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
    
    # Store correctness for each prediction
    correctness_info = []
    
    for pred, gt, full_resp in zip(predicted_answers, ground_truth, full_responses):
        # Initialize correctness data for this prediction
        is_correct = False
        correctness_reason = "incorrect"
        
        # Check answer correctness
        if pred.lower() == gt.lower():
            correct += 1
            is_correct = True
            correctness_reason = "exact_match"
        else:
            gt_str = f'${gt}$' # add $ to make it a valid latex expression
            gt_parsed = parse(gt_str, extraction_mode="first_match")

            pred_parsed = parse(
                pred,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
                )
            
            if pred_parsed is not None:
                try:
                    if verify(gt_parsed, pred_parsed):
                        # if the verification passes, we can verify it but it is not perfect match
                        correct += 1
                        is_correct = True
                        correctness_reason = "symbolic_match"
                except Exception as e:
                    # if the verification fails
                    logger.error(f"Verification error: {e}")
                    logger.error(f"Predicted: {pred}, GT: {gt}")
        
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
        token_length = 0
        if tokenizer is not None:
            tokens = tokenizer.encode(full_resp)
            token_length = len(tokens)
            total_token_length += token_length
        
        # Save correctness info for this prediction
        correctness_info.append({
            "correct": is_correct,
            "reason": correctness_reason,
            "has_think": has_think,
            "has_answer": has_answer,
            "has_boxed": has_box,
            "fully_formatted": has_think and has_answer and has_box,
            "token_length": token_length
        })
    
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
        },
        "correctness_info": correctness_info
    }

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
    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side='left')
    
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
        
    # Save results
    save_results(args, dataset, responses, extracted_answers, metrics)
    
    return metrics

if __name__ == "__main__":
    main()