"""
Evaluate models on openai/gsm8k and HuggingFaceH4/MATH-500
----------------------------------------------------------
* Generate NUM_SAMPLES answers for every question.
* Compute accuracy of the first answer and pass@k.
"""

# ──────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES        = 4          # number of answers generated per question
PASS_AT_K_VALUES   = [1, 4]     # which pass@k metrics to compute
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import re, sys, time, random, logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import yaml

import torch, numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    set_seed, PreTrainedModel, PreTrainedTokenizer
)

from math import comb
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from prompts.prompts import (
    SYSTEM_PROMPT_GSM8K_1_SHOT_EVAL,
    SYSTEM_PROMPT_MATH500_1_SHOT_EVAL,
    SYSTEM_PROMPT_LATENT_REASONER_GSM8K_1_SHOT_EVAL,
    SYSTEM_PROMPT_LATENT_REASONER_MATH500_1_SHOT_EVAL
)
from src.data_process.process_data_eval import prepare_dataset
from src.eval.eval_utils import save_results
from src.latent_reasoner.model import LatentReasoner
from src.train.utils import load_config, setup_special_tokens

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def answer_is_correct(pred: str, gt: str) -> bool:
    """Exact or symbolic match between *one* prediction and ground truth."""
    if pred.lower() == gt.lower():
        return True
    gt_parsed = parse(f'${gt}$', extraction_mode="first_match")
    pred_parsed = parse(
        pred,
        extraction_config=[ LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False,
                basic_latex=True, equations=True,
                boxed="all", units=True,
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )],
        extraction_mode="first_match"
    )
    if pred_parsed is None:
        return False
    try:
        return verify(gt_parsed, pred_parsed)
    except Exception:
        return False


def compute_accuracy_first(pred_first: List[str], gt: List[str]) -> float:
    """Accuracy of the *first* prediction."""
    correctness_list = [answer_is_correct(p, g) for p, g in zip(pred_first, gt)]
    accuracy = sum(correctness_list) / len(correctness_list)
    return accuracy, correctness_list


def compute_pass_at_k(pred_lists: List[List[str]], gt: List[str], k: int) -> float:
    """
    Correct pass@k implementation.

    For each problem i:
        n = NUM_SAMPLES          (constant)
        c_i = # correct samples among the n
        contribution = 1 - C(n - c_i, k) / C(n, k)

    The metric is the mean of contributions.
    """
    n = len(pred_lists[0])                # assume every problem has the same n
    if k > n:
        raise ValueError(f"pass@k needs k ≤ n (k={k}, n={n})")

    scores = []
    for preds, truth in zip(pred_lists, gt):
        # count how many of the n predictions are correct
        c_i = sum(answer_is_correct(p, truth) for p in preds)
        if c_i == 0:
            scores.append(0.0)
        else:
            scores.append(1.0 - comb(n - c_i, k) / comb(n, k))
    return sum(scores) / len(scores) if scores else 0.0


def format_prompt(question: str, dataset_name: str, is_latent_reasoner: bool) -> str:
    """Return a single string prompt (few-shot) for a question."""
    if dataset_name == "openai/gsm8k":
        if is_latent_reasoner:
            system_prompt = SYSTEM_PROMPT_LATENT_REASONER_GSM8K_1_SHOT_EVAL
        else:
            system_prompt = SYSTEM_PROMPT_GSM8K_1_SHOT_EVAL
    else:
        if is_latent_reasoner:
            system_prompt = SYSTEM_PROMPT_LATENT_REASONER_MATH500_1_SHOT_EVAL
        else:
            system_prompt = SYSTEM_PROMPT_MATH500_1_SHOT_EVAL
    return f"{system_prompt}\nUser:{question}\nAssistant:"


def extract_answer_from_response(response: str) -> str:
    """Return content from \boxed{…} in <answer> tag."""
    tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if not tag_match:
        return ""
    answer_block = tag_match.group(1)
    box_match = re.search(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", answer_block,
                          re.DOTALL)
    return (box_match.group(1).strip() if box_match else "")


# Generation routine that returns *all* answers 
def generate_responses_multi(
    cfg,
    model,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    batch_size: int,
    max_length: int,
    temperature: float,
    top_p: float,
    num_samples: int = NUM_SAMPLES,
) -> List[List[str]]:
    """
    Generate `num_samples` responses per prompt.
    Returns: List (len = #questions) of List[str] (len = num_samples)
    """
    model.eval()
    all_out: List[List[str]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating responses", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True).to(model.device)

        # replicate each row num_samples times *inside* the batch
        input_ids = inputs["input_ids"].repeat_interleave(num_samples, dim=0)
        attention = inputs["attention_mask"].repeat_interleave(num_samples, dim=0)

        with torch.no_grad():
            if cfg["model"]["is_latent_reasoner"]:
                # Latent Reasoner model
                outputs, _ = model.generate(
                    input_ids,
                    attention_mask=attention,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1,         # already replicated manually
                    num_latent_steps=cfg["model"]["num_latent_steps"]
                )
            else:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0.0,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=1,         # already replicated manually
                )

        # group back into per-question lists
        for i in range(len(batch_prompts)):
            start_idx = i * num_samples
            samples = outputs[start_idx : start_idx + num_samples]
            if cfg["model"]["is_latent_reasoner"]:
                # Latent Reasoner model returns only the completion token ids
                decoded = [
                    tokenizer.decode(o, skip_special_tokens=True).strip()
                    for o in samples
                ]
            else:
                decoded = [
                    tokenizer.decode(o[input_ids.shape[1]:], skip_special_tokens=True).strip()
                    for o in samples
                ]
            all_out.append(decoded)

    return all_out


def eval_model(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(config_path)
    seed = cfg["seed"]
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # dataset
    dataset = prepare_dataset(cfg["dataset"]["dataset"], cfg["dataset"]["split"], cfg["dataset"]["num_examples"])
    logger.info(f"Evaluating {cfg["model"]["base_model_name_or_path"]} on {cfg["dataset"]["dataset"]}")

    # Initialize model
    if cfg["model"]["is_latent_reasoner"]:
        # For latent reasoner models, use the LatentReasoner class
        model = LatentReasoner.from_pretrained(cfg["model"]["base_model_name_or_path"]).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model_name_or_path"]).to(device)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name_or_path"],
                                              padding_side="left")
    # Setup special tokens
    if cfg["model"]["is_sft_version"]:
        model, tokenizer = setup_special_tokens(
            model=model, 
            tokenizer=tokenizer,
            is_latent_reasoner=cfg["model"]["is_latent_reasoner"]
        )

    # prompts
    prompts = [format_prompt(q, cfg["dataset"]["dataset"], cfg["model"]["is_latent_reasoner"]) for q in dataset["questions"]]

    # ── generate answers ────────────────────────────────────────────────────
    start = time.time()
    responses_multi = generate_responses_multi(
        cfg=cfg,
        model=model, 
        tokenizer=tokenizer, 
        prompts=prompts,
        batch_size=cfg["generation"]["batch_size"],
        max_length=cfg["generation"]["max_length"],
        temperature=cfg["generation"]["temperature"],
        top_p=cfg["generation"]["top_p"],
        num_samples=NUM_SAMPLES,
    )
    generation_time = time.time() - start
    logger.info(f"Generated {NUM_SAMPLES} answers per question "
                f"in {generation_time:.1f}s")

    # extract boxed answers
    extracted_multi = [
        [extract_answer_from_response(r) for r in resp_list]
        for resp_list in responses_multi
    ]

    # first answer lists (for backward-compat accuracy + format checks)
    first_responses = [resp_list[0] for resp_list in responses_multi]
    first_extracted_answers_in_response = [ans_list[0]  for ans_list  in extracted_multi]

    # ── compute metrics ─────────────────────────────────────────────────────
    accuracy_first, correctness_list = compute_accuracy_first(first_extracted_answers_in_response, dataset["answers"])
    metrics = {
        "accuracy":  accuracy_first,
        "num_samples": NUM_SAMPLES,
        "generation_time": generation_time,
        "generation_time_per_question": generation_time / (len(dataset["questions"]) * NUM_SAMPLES)
    }
    
    # pass@k
    for k in PASS_AT_K_VALUES:
        metrics[f"pass@{k}"] = compute_pass_at_k(extracted_multi,
                                                 dataset["answers"], k)

    logger.info(f"Evaluation results:\n{metrics}")

    save_results(cfg, dataset, first_responses, first_extracted_answers_in_response, correctness_list, metrics, tokenizer)


def parse_args():
    parser = argparse.ArgumentParser(description="Evalulation of a model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="src/configs/latent_reasoner_eval.yaml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    eval_model(cli_args.config_path)
