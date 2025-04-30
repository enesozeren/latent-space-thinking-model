from typing import List, Dict, Any, Union, Optional, Tuple
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

def prepare_gsm8k_dataset(split: str = "test", num_examples: Optional[int] = None):
    """Load GSM8K dataset and prepare for evaluation."""
    dataset = load_dataset("gsm8k", "main")
    
    # Get specified split
    ds = dataset[split]
    
    # Limit to num_examples if specified
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    
    # Extract questions and answers
    questions = ds["question"]
    answers = [_extract_gsm8k_answer(ans) for ans in ds["answer"]]
    
    logger.info(f"Loaded {len(questions)} examples from GSM8K {split} split")
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

def prepare_math500_dataset(split: str = "test", num_examples: Optional[int] = None):
    """Load MATH-500 dataset and prepare for evaluation.
    
    The MATH-500 dataset has questions in the 'problem' field and answers in the 'answer' field.
    Note that it only has a 'test' split.
    """
    if split != "test":
        logger.warning(f"MATH-500 dataset only has a 'test' split. Using 'test' instead of requested '{split}'.")
        split = "test"
        
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    
    # Get test split (the only one available)
    ds = dataset[split]
    
    # Limit to num_examples if specified
    if num_examples is not None:
        ds = ds.select(range(min(num_examples, len(ds))))
    
    # Extract questions and answers
    questions = ds["problem"]
    answers = ds["answer"]
    
    logger.info(f"Loaded {len(questions)} examples from MATH-500 dataset")
    return {"questions": questions, "answers": answers}