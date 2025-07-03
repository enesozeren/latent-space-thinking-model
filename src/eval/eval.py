"""
Evaluate models on openai/gsm8k and HuggingFaceH4/MATH-500
----------------------------------------------------------
* Generate NUM_SAMPLES answers for every question.
* Compute accuracy of the first answer and pass@k.

Updated for multi-GPU execution with **torchrun --standalone --nproc_per_node=4**.
Changes are limited to:
  • imports for distributed support
  • distributed initialisation & device selection
  • per-rank dataset slicing
  • gathering predictions/answers for metric computation and saving on rank-0
  • final cleanup of the process group
Everything else is untouched.
"""

# ──────────────────────────────────────────────────────────────────────────────
NUM_SAMPLES        = 4          # number of answers generated per question
PASS_AT_K_VALUES   = [1, 4]     # which pass@k metrics to compute
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import re, sys, time, random, logging, os                 # ← added os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import yaml

import torch, numpy as np
import torch.distributed as dist                          # ← distributed import
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
from src.train.utils import load_config, setup_latent_tokens

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


def clean_decoded_text(text: str, tokenizer: PreTrainedTokenizer) -> str:
    """Remove specific special tokens from decoded text while preserving others."""
    # Get the string representations of special tokens we want to remove
    tokens_to_remove = [tokenizer.pad_token, tokenizer.eos_token]
    
    # Remove the tokens from the text
    cleaned_text = text
    for token in tokens_to_remove:
        if token:
            cleaned_text = cleaned_text.replace(token, '')
    
    return cleaned_text.strip()


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
    # ── distributed initialisation ───────────────────────────────────────────
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp   = world_size > 1
    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        rank       = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ─────────────────────────────────────────────────────────────────────────

    cfg = load_config(config_path)
    seed = cfg["seed"]
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # dataset (created on every rank, then sliced)
    dataset = prepare_dataset(cfg["dataset"]["dataset"],
                              cfg["dataset"]["split"],
                              cfg["dataset"]["num_examples"])

    if rank == 0:
        logger.info(f"Evaluating {cfg['model']['base_model_name_or_path']} "
                    f"on {cfg['dataset']['dataset']} "
                    f"with world_size={world_size}")

    # per-rank slicing
    questions_rank = dataset["questions"][rank::world_size]
    answers_rank   = dataset["answers"][rank::world_size]

    # Initialize model
    if cfg["model"]["is_latent_reasoner"]:
        model = LatentReasoner.from_pretrained(
            cfg["model"]["base_model_name_or_path"]
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"]["base_model_name_or_path"]
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model_name_or_path"],
        padding_side="left"
    )
    
    # Setup special tokens
    model, tokenizer = setup_latent_tokens(
        model=model, 
        tokenizer=tokenizer,
        is_latent_reasoner=cfg["model"]["is_latent_reasoner"]
    )

    # prompts
    prompts = [format_prompt(q, cfg["dataset"]["dataset"],
                             cfg["model"]["is_latent_reasoner"])
               for q in questions_rank]

    # print example prompt (rank-0 only)
    if rank == 0 and prompts:
        logger.info(f"Example prompt:\n{prompts[0]}")

    # ── generate answers ────────────────────────────────────────────────────
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


    # extract boxed answers
    extracted_multi = [
        [extract_answer_from_response(r) for r in resp_list]
        for resp_list in responses_multi
    ]

    # first answer lists
    first_responses = [resp_list[0] for resp_list in responses_multi]

    # ── gather predictions & answers ────────────────────────────────────────
    if use_ddp:
        # gather lists from all ranks
        gathered_preds   = [None] * world_size
        gathered_first   = [None] * world_size
        gathered_answers = [None] * world_size

        dist.all_gather_object(gathered_preds, extracted_multi)
        dist.all_gather_object(gathered_first, first_responses)
        dist.all_gather_object(gathered_answers, answers_rank)

        # rank-0 concatenates
        if rank == 0:
            extracted_multi = [p for part in gathered_preds for p in part]
            first_responses = [f for part in gathered_first for f in part]
            answers_all     = [a for part in gathered_answers for a in part]
    else:
        answers_all = answers_rank

    # ── compute & save metrics (rank-0) ─────────────────────────────────────
    if rank == 0:
        first_extracted_answers = [ans_list[0] for ans_list in extracted_multi]

        accuracy_first, correctness_list = compute_accuracy_first(
            first_extracted_answers, answers_all
        )
        metrics = {
            "accuracy":  accuracy_first,
            "num_samples": NUM_SAMPLES
        }
        # pass@k
        for k in PASS_AT_K_VALUES:
            metrics[f"pass@{k}"] = compute_pass_at_k(
                extracted_multi, answers_all, k
            )

        logger.info(f"Evaluation results:\n{metrics}")

        save_results(cfg, {
                "questions": dataset["questions"],
                "answers":   dataset["answers"]
            },
            first_responses,
            first_extracted_answers,
            correctness_list,
            metrics,
            tokenizer
        )

    # ensure all ranks finish before teardown
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


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
