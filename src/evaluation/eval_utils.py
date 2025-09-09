"""
Shared utilities for evaluation: epsilon masking, option scoring, generation,
and common metrics (SQuAD EM/F1, GLUE metrics).
"""

from typing import List, Tuple, Optional, Dict
import math
import random
import re

import numpy as np
import torch
import torch.nn.functional as F


# Punctuation regex and function words set (aligned with Winograd evaluator)
_PUNCT_RE = re.compile(r"^[\\.,!?;:\-()\[\]{}\'\"…]+$")
_FUNCTION_WORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with","as","is","are","was","were","be","been","being","that","this","those","these","it","its","he","she","they","them","his","her","their","we","you","i","not","no","do","does","did","from","because","so","than","too","very","can","could","should","would","will","may","might"
}


def _is_function_word_token(token: str) -> bool:
    clean_token = token.replace("##", "").replace("Ġ", "")
    clean = re.sub(r"[^a-z]", "", clean_token.lower())
    return clean in _FUNCTION_WORDS


def _is_punctuation_token(token: str) -> bool:
    clean_token = token.replace("##", "").replace("Ġ", "")
    return bool(_PUNCT_RE.match(clean_token))


def apply_epsilon_masking(tokenizer, text: str, epsilon: float, seed: Optional[int] = None) -> str:
    """
    Apply epsilon-masking using the given tokenizer.

    - Preserves function words and punctuation
    - Masks other tokens with probability epsilon using tokenizer.mask_token if present
    """
    if epsilon <= 0.0:
        return text

    if seed is not None:
        random.seed(seed)

    tokens = tokenizer.tokenize(text)
    masked_tokens: List[str] = []
    for token in tokens:
        if _is_function_word_token(token) or _is_punctuation_token(token):
            masked_tokens.append(token)
        else:
            if random.random() < epsilon:
                mask_token = getattr(tokenizer, "mask_token", None) or "<MASK>"
                masked_tokens.append(mask_token)
            else:
                masked_tokens.append(token)

    return tokenizer.convert_tokens_to_string(masked_tokens)


@torch.no_grad()
def score_option(tokenizer, model, device: str, prompt: str, option: str) -> Tuple[float, float, int]:
    """
    Compute total and average log-probability for `option` given `prompt`.
    Returns (total_logprob, avg_logprob, option_token_count).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_text = f"{prompt} {option}" if len(prompt) > 0 else option
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = full.input_ids.to(device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, L, V]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]

    prompt_len = prompt_ids.size(1)
    total_len = input_ids.size(1)
    option_len = max(total_len - prompt_len, 0)
    if option_len == 0:
        return float("-inf"), float("-inf"), 0

    start = max(prompt_len - 1, 0)
    end = total_len - 1
    rng = torch.arange(start, end, device=device)
    option_token_log_probs = log_probs[0, rng, labels[0, rng]]

    total_logprob = option_token_log_probs.sum().item()
    avg_logprob = (total_logprob / float(option_len)) if option_len > 0 else float("-inf")
    return total_logprob, avg_logprob, int(option_len)


@torch.no_grad()
def greedy_generate(tokenizer, model, device: str, prompt: str, max_new_tokens: int = 32, stop_at_newline: bool = True) -> str:
    """
    Greedy generation of up to `max_new_tokens` following the prompt.
    Returns only the generated continuation text (without the prompt).
    """
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = output_ids[0][inputs.input_ids.size(1):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if stop_at_newline:
        nl_idx = generated_text.find("\n")
        if nl_idx >= 0:
            generated_text = generated_text[:nl_idx]
    return generated_text.strip()


# ---------------------- SQuAD metrics ----------------------

_ARTICLES = {"a", "an", "the"}
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = _NON_ALNUM_RE.sub(" ", s)
    s = " ".join(tok for tok in s.split() if tok not in _ARTICLES)
    return s.strip()


def squad_em_f1(prediction: str, ground_truths: List[str]) -> Tuple[float, float]:
    if not ground_truths:
        return 0.0, 0.0
    pred = _normalize_answer(prediction)

    def f1_score(a_pred: str, a_true: str) -> float:
        pred_tokens = a_pred.split()
        true_tokens = _normalize_answer(a_true).split()
        common = {}
        for t in pred_tokens:
            common[t] = min(common.get(t, 0) + (t in true_tokens), 1)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            return float(pred_tokens == true_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(true_tokens)
        return 2 * precision * recall / (precision + recall)

    em = 0.0
    best_f1 = 0.0
    for gt in ground_truths:
        em = max(em, float(pred == _normalize_answer(gt)))
        best_f1 = max(best_f1, f1_score(pred, gt))
    return em, best_f1


# ---------------------- GLUE metrics ----------------------

def accuracy(preds: List[str], golds: List[str]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if p == g)
    return correct / len(preds)


def f1_binary(preds: List[str], golds: List[str], positive_label: str) -> float:
    tp = sum(1 for p, g in zip(preds, golds) if p == positive_label and g == positive_label)
    fp = sum(1 for p, g in zip(preds, golds) if p == positive_label and g != positive_label)
    fn = sum(1 for p, g in zip(preds, golds) if p != positive_label and g == positive_label)
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def matthews_corrcoef_binary(preds: List[str], golds: List[str], positive_label: str) -> float:
    tp = sum(1 for p, g in zip(preds, golds) if p == positive_label and g == positive_label)
    tn = sum(1 for p, g in zip(preds, golds) if p != positive_label and g != positive_label)
    fp = sum(1 for p, g in zip(preds, golds) if p == positive_label and g != positive_label)
    fn = sum(1 for p, g in zip(preds, golds) if p != positive_label and g == positive_label)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return ((tp * tn) - (fp * fn)) / denom


