"""
SQuAD-style evaluation using generation with epsilon-masked contexts.

Dataset format (JSON list):
[
  {
    "id": "q1",
    "context": "...",
    "question": "...",
    "answers": ["answer text", "alternative phrasing"],
    "title": "optional"
  }
]
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
import math

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .eval_utils import apply_epsilon_masking, greedy_generate, squad_em_f1


class SquadEvaluator:
    """
    Evaluates models on SQuAD-style QA using simple generation.
    - Uses epsilon-masked context
    - Computes EM and token-level F1
    """

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, masked_context: str, question: str) -> str:
        return f"Context: {masked_context}\nQuestion: {question}\nAnswer:"

    def evaluate_example(self, example: Dict, epsilon: float = 0.0, seed: Optional[int] = None,
                         max_new_tokens: int = 32) -> Dict:
        context = example.get("context", "")
        question = example.get("question", "")
        answers: List[str] = example.get("answers", [])

        masked_context = apply_epsilon_masking(self.tokenizer, context, epsilon, seed)
        prompt = self._build_prompt(masked_context, question)
        prediction = greedy_generate(self.tokenizer, self.model, self.device, prompt, max_new_tokens=max_new_tokens)

        em, f1 = squad_em_f1(prediction, answers)

        return {
            "id": example.get("id"),
            "epsilon": epsilon,
            "context": context,
            "masked_context": masked_context,
            "question": question,
            "answers": answers,
            "prediction": prediction,
            "em": float(em),
            "f1": float(f1),
            "title": example.get("title")
        }

    def evaluate_all(self, examples: List[Dict], epsilon: float = 0.0, seed: Optional[int] = None,
                     max_new_tokens: int = 32) -> List[Dict]:
        results: List[Dict] = []
        for i, ex in enumerate(examples):
            per_seed = None if seed is None else seed + i
            results.append(self.evaluate_example(ex, epsilon, per_seed, max_new_tokens))
        return results

    def get_performance_metrics(self, results: List[Dict]) -> Dict:
        total = len(results)
        if total == 0:
            return {"total": 0, "em": 0.0, "f1": 0.0, "avg_pred_len": 0.0}
        ems = [r.get("em", 0.0) for r in results]
        f1s = [r.get("f1", 0.0) for r in results]
        pred_lens = [len(str(r.get("prediction", "")).split()) for r in results]
        return {
            "total": total,
            "em": float(np.mean(ems)) if ems else 0.0,
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "avg_pred_len": float(np.mean(pred_lens)) if pred_lens else 0.0,
        }


