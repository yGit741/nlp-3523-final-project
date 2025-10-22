"""
Winograd Schema Challenge evaluation.

This module implements comprehensive evaluation of models on Winograd
Schema Challenge with detailed error analysis.
"""

from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import json
import math
import random
import re
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


class WinogradEvaluator:
    """
    Evaluates models on Winograd Schema Challenge.
    
    Features:
    - Multiple model support (HF hub name or local path)
    - Epsilon masking with simple structural preservation
    - Confidence scoring via average token log-prob
    - Performance metrics and lightweight error analysis
    """
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """
        Initialize Winograd evaluator.
        
        Args:
            model_name: Hugging Face model name or local path
            device: Device to use for evaluation
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

        # Load tokenizer/model (works for both hub names and local paths)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            # GPT-2 style tokenizers often lack pad token; use EOS
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        # Compile simple punctuation regex once
        self._punct_re = re.compile(r"^[\\.,!?;:\-()\[\]{}\'\"…]+$")

        # Fallback minimal function-word list to avoid external downloads
        self._function_words = {
            "the","a","an","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with","as","is","are","was","were","be","been","being","that","this","those","these","it","its","he","she","they","them","his","her","their","we","you","i","not","no","do","does","did","from","because","so","than","too","very","can","could","should","would","will","may","might"
        }

    def _is_function_word(self, token_text: str) -> bool:
        """Check if a space-separated word is a function word (for fallback compatibility)."""
        clean = re.sub(r"[^a-z]", "", token_text.lower())
        return clean in self._function_words

    def _is_punctuation(self, token_text: str) -> bool:
        """Check if a space-separated word is punctuation (for fallback compatibility)."""
        return bool(self._punct_re.match(token_text))

    def _is_function_word_token(self, token: str) -> bool:
        """Check if a tokenizer token represents a function word."""
        # Handle subword tokens (remove ## prefix if present)
        clean_token = token.replace("##", "").replace("Ġ", "")  # Handle BERT/GPT2 style tokens
        clean = re.sub(r"[^a-z]", "", clean_token.lower())
        return clean in self._function_words

    def _is_punctuation_token(self, token: str) -> bool:
        """Check if a tokenizer token represents punctuation."""
        # Handle subword tokens
        clean_token = token.replace("##", "").replace("Ġ", "")
        return bool(self._punct_re.match(clean_token))

    def _apply_epsilon_masking(self, text: str, epsilon: float, seed: Optional[int] = None) -> str:
        """
        Apply ε-masking using proper tokenization that matches the model.
        
        Args:
            text: Input text to mask
            epsilon: Probability of masking each content token (0.0 = no masking, 1.0 = mask all)
            seed: Random seed for reproducibility
            
        Returns:
            Masked text string
        """
        if epsilon <= 0.0:
            return text
        
        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Tokenize using the model's tokenizer
        tokens = self.tokenizer.tokenize(text)
        
        masked_tokens = []
        for token in tokens:
            # Preserve function words and punctuation
            if self._is_function_word_token(token) or self._is_punctuation_token(token):
                masked_tokens.append(token)
            else:
                # Apply epsilon masking to content words
                if random.random() < epsilon:
                    # Use the tokenizer's mask token if available, otherwise fallback
                    mask_token = getattr(self.tokenizer, 'mask_token', None) or "<MASK>"
                    masked_tokens.append(mask_token)
                else:
                    masked_tokens.append(token)
        
        # Convert back to text using tokenizer's method
        return self.tokenizer.convert_tokens_to_string(masked_tokens)

    def _apply_simple_epsilon_masking(self, text: str, epsilon: float) -> str:
        """
        Legacy method for backward compatibility. Use _apply_epsilon_masking instead.
        """
        import warnings
        warnings.warn(
            "_apply_simple_epsilon_masking is deprecated. Use _apply_epsilon_masking instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._apply_epsilon_masking(text, epsilon)

    @torch.no_grad()
    def _score_option(self, prompt: str, option: str) -> Tuple[float, float, int]:
        """
        Return (total_logprob, avg_logprob, option_token_count) for the option continuation.
        """
        # Tokenize prompt and full text separately to find the split
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        full_text = f"{prompt} {option}" if len(prompt) > 0 else option
        full = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = full.input_ids.to(self.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits  # [1, L, V]
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # predict tokens 1..L-1
        labels = input_ids[:, 1:]  # ground-truth next tokens

        # Determine the indices corresponding to the option tokens
        prompt_len = prompt_ids.size(1)
        total_len = input_ids.size(1)
        option_len = max(total_len - prompt_len, 0)
        if option_len == 0:
            return float("-inf"), float("-inf"), 0

        # Labels start at position 0 (predicting token index 1). Option labels start at (prompt_len-1)
        start = max(prompt_len - 1, 0)
        end = total_len - 1  # inclusive for labels indexing (since labels length is total_len-1)

        # Gather log probs for the true labels in the option range
        rng = torch.arange(start, end, device=self.device)
        option_token_log_probs = log_probs[0, rng, labels[0, rng]]  # shape [option_len]

        total_logprob = option_token_log_probs.sum().item()
        avg_logprob = (total_logprob / float(option_len)) if option_len > 0 else float("-inf")
        return total_logprob, avg_logprob, int(option_len)

    def evaluate_schema(self, schema: Dict, epsilon: float = 0.0, seed: Optional[int] = None) -> Dict:
        """
        Evaluate a single Winograd schema.
        
        Args:
            schema: Winograd schema dictionary containing text, question, options, answer
            epsilon: Probability of masking content tokens (0.0 = no masking, 1.0 = mask all)
            seed: Random seed for reproducible masking
            
        Returns:
            Detailed evaluation results
        """
        text = schema.get("text", "")
        question = schema.get("question", "")
        options: List[str] = schema.get("options", [])
        correct_answer = schema.get("answer")

        masked_text = self._apply_epsilon_masking(text, epsilon, seed)
        prompt = f"{masked_text} {question} Answer:"

        option_details = []
        option_scores = []
        for opt in options:
            tot_lp, avg_lp, opt_len = self._score_option(prompt, opt)
            option_details.append({
                "option": opt,
                "total_logprob": tot_lp,
                "avg_logprob": avg_lp,
                "length": opt_len
            })
            option_scores.append(avg_lp)

        predicted_idx = int(np.argmax(option_scores)) if option_scores else -1
        predicted_answer = options[predicted_idx] if 0 <= predicted_idx < len(options) else None
        is_correct = (predicted_answer == correct_answer)

        # Confidence as exp(avg_logprob) of the chosen option
        confidence = math.exp(option_scores[predicted_idx]) if predicted_idx >= 0 else 0.0

        return {
            "schema_id": schema.get("id"),
            "original_text": text,
            "masked_text": masked_text,
            "epsilon": epsilon,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": bool(is_correct),
            "option_scores": option_scores,
            "option_details": option_details,
            "confidence": confidence,
            "difficulty": schema.get("difficulty"),
            "reasoning": schema.get("reasoning")
        }
    
    def evaluate_all_schemas(self, schemas: List[Dict], epsilon: float = 0.0, seed: Optional[int] = None) -> List[Dict]:
        """
        Evaluate all schemas in dataset.
        
        Args:
            schemas: List of Winograd schema dictionaries
            epsilon: Probability of masking content tokens (0.0 = no masking, 1.0 = mask all)
            seed: Random seed for reproducible masking
            
        Returns:
            List of evaluation results
        """
        results: List[Dict] = []
        for i, schema in enumerate(schemas):
            # Use a deterministic seed per schema if base seed is provided
            schema_seed = None if seed is None else seed + i
            results.append(self.evaluate_schema(schema, epsilon, schema_seed))
        return results
    
    def evaluate_model(self, model_path: Union[str, Path], schemas: List[Dict], 
                      epsilon: float = 0.0, seed: Optional[int] = None) -> Dict:
        """
        Evaluate a (potentially different) model on Winograd schemas.
        
        Args:
            model_path: HF model name or local path to model
            schemas: List of Winograd schema dictionaries
            epsilon: Probability of masking content tokens (0.0 = no masking, 1.0 = mask all)
            seed: Random seed for reproducible masking
            
        Returns:
            Evaluation results with metrics and detailed results
        """
        model_path = str(model_path)
        if model_path != self.model_name:
            # Reload model/tokenizer for the requested path
            self.model_name = model_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

        detailed = self.evaluate_all_schemas(schemas, epsilon, seed)
        metrics = self.get_performance_metrics(detailed)
        return {"metrics": metrics, "detailed": detailed}
    
    def get_performance_metrics(self, results: List[Dict]) -> Dict:
        """
        Calculate basic performance metrics from results.
        """
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct"))
        accuracy = (correct / total) if total > 0 else 0.0

        # Accuracy by difficulty
        difficulties = {}
        for r in results:
            diff = r.get("difficulty", "unknown")
            d = difficulties.setdefault(diff, {"total": 0, "correct": 0})
            d["total"] += 1
            d["correct"] += 1 if r.get("is_correct") else 0
        acc_by_diff = {k: (v["correct"] / v["total"]) if v["total"] > 0 else 0.0 for k, v in difficulties.items()}

        confidences = [r.get("confidence", 0.0) for r in results]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        avg_conf_correct = float(np.mean([r.get("confidence", 0.0) for r in results if r.get("is_correct")])) if total > 0 else 0.0
        avg_conf_errors = float(np.mean([r.get("confidence", 0.0) for r in results if not r.get("is_correct")])) if total > 0 else 0.0

        return {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "accuracy_by_difficulty": acc_by_diff,
            "avg_confidence": avg_conf,
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_errors": avg_conf_errors,
        }
    
    def save_results(self, results: Dict, output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def load_results(self, input_path: Union[str, Path]) -> Dict:
        input_path = Path(input_path)
        with input_path.open("r", encoding="utf-8") as f:
            return json.load(f)
