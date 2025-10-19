"""
GLUE-style evaluation via option scoring prompts.

Supported tasks in this initial pass: SST-2, CoLA, MRPC, QQP, RTE, QNLI, MNLI.

Input dataset format: JSON list of examples with task-dependent fields.
Examples:
  - SST-2/CoLA: {"id": "ex1", "sentence": "...", "label": "positive"}
  - MRPC/QQP: {"id": "ex2", "sentence1": "...", "sentence2": "...", "label": "paraphrase"}
  - RTE/QNLI/MNLI: {"id": "ex3", "premise": "...", "hypothesis": "...", "label": "entailment"}
"""

from typing import List, Dict, Optional, Union, Tuple
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .eval_utils import apply_epsilon_masking, score_option, accuracy, f1_binary, matthews_corrcoef_binary


GLUE_LABELS = {
    "SST-2": ["positive", "negative"],
    "CoLA": ["acceptable", "unacceptable"],
    "MRPC": ["paraphrase", "not paraphrase"],
    "QQP": ["paraphrase", "not paraphrase"],
    "RTE": ["entailment", "not entailment"],
    "QNLI": ["entailment", "not entailment"],
    "MNLI": ["entailment", "neutral", "contradiction"],
}


class GlueEvaluator:
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _prompt_sst2(self, sentence: str) -> str:
        return f"Task: sentiment classification.\nSentence: {sentence}\nOptions: positive, negative.\nAnswer:"

    def _prompt_cola(self, sentence: str) -> str:
        return f"Task: linguistic acceptability.\nSentence: {sentence}\nOptions: acceptable, unacceptable.\nAnswer:"

    def _prompt_pair(self, task: str, s1: str, s2: str) -> str:
        if task in ("MRPC", "QQP"):
            return (
                f"Task: paraphrase detection.\n"
                f"Sentence1: {s1}\nSentence2: {s2}\n"
                f"Options: paraphrase, not paraphrase.\nAnswer:"
            )
        if task in ("RTE", "QNLI"):
            return (
                f"Task: natural language inference.\n"
                f"Premise: {s1}\nHypothesis: {s2}\n"
                f"Options: entailment, not entailment.\nAnswer:"
            )
        if task == "MNLI":
            return (
                f"Task: natural language inference.\n"
                f"Premise: {s1}\nHypothesis: {s2}\n"
                f"Options: entailment, neutral, contradiction.\nAnswer:"
            )
        raise ValueError(f"Unsupported pairwise task: {task}")

    def _mask_text(self, text: str, epsilon: float, seed: Optional[int]) -> str:
        return apply_epsilon_masking(self.tokenizer, text, epsilon, seed)

    def evaluate_example(self, task: str, example: Dict, epsilon: float = 0.0, seed: Optional[int] = None) -> Dict:
        labels = GLUE_LABELS.get(task)
        if not labels:
            raise ValueError(f"Unsupported GLUE task: {task}")

        if task in ("SST-2", "CoLA"):
            sent = example.get("sentence", "")
            m_sent = self._mask_text(sent, epsilon, seed)
            prompt = self._prompt_sst2(m_sent) if task == "SST-2" else self._prompt_cola(m_sent)
        elif task in ("MRPC", "QQP"):
            s1 = example.get("sentence1", "")
            s2 = example.get("sentence2", "")
            m1 = self._mask_text(s1, epsilon, seed)
            m2 = self._mask_text(s2, epsilon, None if seed is None else seed + 1)
            prompt = self._prompt_pair(task, m1, m2)
        elif task in ("RTE", "QNLI", "MNLI"):
            premise = example.get("premise", "")
            hypothesis = example.get("hypothesis", "")
            mp = self._mask_text(premise, epsilon, seed)
            mh = self._mask_text(hypothesis, epsilon, None if seed is None else seed + 1)
            prompt = self._prompt_pair(task, mp, mh)
        else:
            raise ValueError(f"Unsupported task: {task}")

        option_scores: List[float] = []
        for opt in labels:
            _, avg_lp, _ = score_option(self.tokenizer, self.model, self.device, prompt, opt)
            option_scores.append(avg_lp)
        pred_idx = int(np.argmax(option_scores)) if option_scores else -1
        prediction = labels[pred_idx] if 0 <= pred_idx < len(labels) else None

        return {
            "id": example.get("id"),
            "epsilon": epsilon,
            "task": task,
            "label": example.get("label"),
            "prediction": prediction,
            "option_scores": option_scores,
        }

    def evaluate_all(self, task: str, examples: List[Dict], epsilon: float = 0.0, seed: Optional[int] = None) -> List[Dict]:
        results: List[Dict] = []
        for i, ex in enumerate(examples):
            per_seed = None if seed is None else seed + i
            results.append(self.evaluate_example(task, ex, epsilon, per_seed))
        return results

    def get_performance_metrics(self, task: str, results: List[Dict]) -> Dict:
        total = len(results)
        labels = [r.get("label") for r in results]
        preds = [r.get("prediction") for r in results]
        acc = accuracy(preds, labels)

        metrics: Dict[str, float] = {"total": total, "accuracy": acc}
        if task in ("MRPC", "QQP"):
            metrics["f1"] = f1_binary(preds, labels, positive_label="paraphrase")
        if task == "CoLA":
            metrics["mcc"] = matthews_corrcoef_binary(preds, labels, positive_label="acceptable")
        return metrics


