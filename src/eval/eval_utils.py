import os
import re
import json
import yaml
from datetime import datetime
from pathlib import Path
import wandb
import logging

logger = logging.getLogger(__name__)

def get_format_info(response, tokenizer, is_latent_reasoner):
    """Extract format information from the model response."""
    
    if is_latent_reasoner:
        # full_pattern = re.compile(
        #     r"^<\|start-latent\|>(?:<\|latent\|>)*<\|end-latent\|><think>.*?</think>\s*<answer>.*?\\boxed\{.*?\}.*?</answer>$",
        #     re.DOTALL
        # )
        full_pattern = re.compile(
            r"^<\|start-latent\|>(?:<\|latent\|>)*<\|end-latent\|><answer>.*?\\boxed\{.*?\}.*?</answer>$",
            re.DOTALL
        )
    else:
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

def save_results(cfg, dataset, first_responses, first_extracted_answers_in_response, correctness_list, metrics, tokenizer):
    """Save evaluation results to output directory."""
    is_latent_reasoner = cfg["model"]["is_latent_reasoner"]
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(cfg["model"]["base_model_name_or_path"]).name
    dataset_name = cfg["dataset"]["dataset"]
    # Include dataset name in output folder
    output_dir = Path(cfg["logs"]["output_dir"]) / f"{model_name}_{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create examples list for both local save and wandb
    examples_list = []
    examples_path = output_dir / "examples.jsonl"
    
    with open(examples_path, "w") as f:
        for i, (question, gt_answer, response, extracted_answer, is_correct) in enumerate(
            zip(dataset["questions"], dataset["answers"], first_responses, first_extracted_answers_in_response, correctness_list)
        ):
            
            has_boxed, fully_formatted, token_length = get_format_info(response, tokenizer, is_latent_reasoner)
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
    metrics["std_token_length"] = (sum((x - metrics["avg_token_length"]) ** 2 for x in token_length_list) / len(token_length_list)) ** 0.5
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Log to W&B if not disabled
    if not cfg["logs"]["no_wandb"]:
        # Initialize wandb if not already running
        run_name = cfg["logs"]["wandb_run_name"] or f"{model_name}_{dataset_name}_eval"
        run = wandb.init(
            project=cfg["logs"]["wandb_project"], 
            entity="lmu-thesis-team",
            name=run_name, 
            config=cfg,
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
