import os
import re
import json
import yaml
from datetime import datetime
from pathlib import Path
import wandb
import logging

logger = logging.getLogger(__name__)

def get_format_info(response, tokenizer):
    """Extract format information from the model response."""
    
    full_pattern = re.compile(
        r"^<think>.*?</think>\s*<answer>.*?\\boxed\{.*?\}.*?</answer>$",
        re.DOTALL
    )
    fully_formatted = bool(full_pattern.match(response.strip()))

    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    has_boxed = bool(re.search(boxed_pattern, response))

    # Tokenize the response
    tokenized_response = tokenizer(response, return_tensors="pt")
    token_length = len(tokenized_response["input_ids"][0])
    
    return has_boxed, fully_formatted, token_length

def save_results(args, dataset, first_responses, first_extracted_answers_in_response, correctness_list, metrics, tokenizer):
    """Save evaluation results to output directory."""
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_name_or_path).name
    # Include dataset name in output folder
    output_dir = Path(args.output_dir) / f"{model_name}_{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create examples list for both local save and wandb
    examples_list = []
    examples_path = output_dir / "examples.jsonl"
    
    with open(examples_path, "w") as f:
        for i, (question, gt_answer, response, extracted_answer, is_correct) in enumerate(
            zip(dataset["questions"], dataset["answers"], first_responses, first_extracted_answers_in_response, correctness_list)
        ):
            
            has_boxed, fully_formatted, token_length = get_format_info(response, tokenizer)
            example = {
                "id": i,
                "question": question,
                "ground_truth": gt_answer,
                "model_response": response,
                "extracted_answer": extracted_answer,
                "correct": is_correct,
                "format_info": {
                    "has_boxed": has_boxed,
                    "fully_formatted": fully_formatted
                },
                "token_length": token_length
            }
            examples_list.append(example)
            f.write(json.dumps(example) + "\n")
    
    # Calculate the has_boxed, fully_formatted and token length metrics
    has_boxed_count = sum(1 for ex in examples_list if ex["format_info"]["has_boxed"])
    fully_formatted_count = sum(1 for ex in examples_list if ex["format_info"]["fully_formatted"])
    token_length_list = [ex["token_length"] for ex in examples_list]
    total_count = len(examples_list)
    metrics["has_boxed"] = has_boxed_count / total_count
    metrics["fully_formatted"] = fully_formatted_count / total_count
    metrics["avg_token_length"] = sum(token_length_list) / len(token_length_list)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

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
        columns = ["id", "question", "ground_truth", "model_response", "extracted_answer", 
                  "correct", "has_boxed", "fully_formatted"]
        example_table = wandb.Table(columns=columns)
        
        # Add each example to the table
        for ex in examples_list:
            example_table.add_data(ex["id"], ex["question"], ex["ground_truth"], 
                                  ex["model_response"], ex["extracted_answer"], ex["correct"],
                                  ex["format_info"]["has_boxed"], ex["format_info"]["fully_formatted"])
        
        # Log table
        wandb.log({"examples": example_table})
        
        # Upload JSON files to wandb
        wandb.save(str(metrics_path))
        wandb.save(str(examples_path))
        wandb.save(str(config_path))
        
        logger.info(f"Results saved to {output_dir} and logged to W&B")
    else:
        logger.info(f"Results saved to {output_dir} (W&B logging disabled)")
        
    return output_dir
