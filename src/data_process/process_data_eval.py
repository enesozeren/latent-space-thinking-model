from typing import Optional
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

def prepare_dataset(dataset_name: str, split: str = "test", num_examples: Optional[int] = None):
    """Load the dataset and prepare for evaluation."""
    
    if dataset_name == "openai/gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    elif dataset_name == "HuggingFaceH4/MATH-500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")
    
    # Get specified split
    ds = dataset[split]
    
    # Limit to num_examples if specified
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    
    # Extract questions and answers
    if dataset_name == "openai/gsm8k":
        questions = ds["question"]
        answers = [_extract_gsm8k_answer(ans) for ans in ds["answer"]]
    elif dataset_name == "HuggingFaceH4/MATH-500":
        questions = ds["problem"]
        answers = ds["answer"]
    
    logger.info(f"Loaded {len(questions)} examples from {dataset_name} {split} split")

    return {"questions": questions, "answers": answers}

def _extract_gsm8k_answer(raw_answer: str) -> str:
    """
    Extract the canonical short answer from a GSM8K solution string.

    GSM8K places the final numeric answer after the delimiter '####', e.g.
        "... reasoning ...\n#### 24"
    We take everything after the *last* occurrence of that delimiter,
    strip whitespace, and drop a trailing period if present.
    """
    if "####" in raw_answer:
        answer = raw_answer.split("####")[-1]
    else:
        answer = raw_answer
    answer = answer.strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    return answer