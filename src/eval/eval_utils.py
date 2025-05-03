import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import wandb
import logging

logger = logging.getLogger(__name__)

def safe_float_convert(value):
    """Safely convert a value to float, returning None if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


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
    
    # Use correctness information from metrics if available
    correctness_info = metrics.get("correctness_info", [])
    
    with open(examples_path, "w") as f:
        for i, (question, gt_answer, response, extracted_answer) in enumerate(
            zip(dataset["questions"], dataset["answers"], responses, extracted_answers)
        ):
            # Get correctness info for this example if available
            is_correct = False
            correctness_reason = "incorrect"
            has_think = False
            has_answer = False
            has_boxed = False
            fully_formatted = False
            token_length = 0
            
            if i < len(correctness_info):
                is_correct = correctness_info[i]["correct"]
                correctness_reason = correctness_info[i]["reason"]
                has_think = correctness_info[i]["has_think"]
                has_answer = correctness_info[i]["has_answer"]
                has_boxed = correctness_info[i]["has_boxed"]
                fully_formatted = correctness_info[i]["fully_formatted"]
                token_length = correctness_info[i]["token_length"]
            
            example = {
                "id": i,
                "question": question,
                "ground_truth": gt_answer,
                "model_response": response,
                "extracted_answer": extracted_answer,
                "correct": is_correct,
                "correctness_reason": correctness_reason,
                "format_info": {
                    "has_think": has_think,
                    "has_answer": has_answer,
                    "has_boxed": has_boxed,
                    "fully_formatted": fully_formatted
                },
                "token_length": token_length
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
        columns = ["id", "question", "ground_truth", "model_response", "extracted_answer", 
                  "correct", "correctness_reason", "has_think", "has_answer", "has_boxed", "fully_formatted"]
        example_table = wandb.Table(columns=columns)
        
        # Add each example to the table
        for ex in examples_list:
            example_table.add_data(ex["id"], ex["question"], ex["ground_truth"], 
                                  ex["model_response"], ex["extracted_answer"], ex["correct"],
                                  ex["correctness_reason"], ex["format_info"]["has_think"],
                                  ex["format_info"]["has_answer"], ex["format_info"]["has_boxed"],
                                  ex["format_info"]["fully_formatted"])
        
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
