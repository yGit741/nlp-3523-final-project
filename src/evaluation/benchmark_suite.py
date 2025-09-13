"""
Comprehensive benchmark suite for model evaluation.

This module provides a unified interface for evaluating models
on multiple benchmarks including Winograd and simple custom tasks.
"""

from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import json
import time

from .winograd_evaluator import WinogradEvaluator
from .squad_evaluator import SquadEvaluator
from .glue_evaluator import GlueEvaluator


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for model evaluation.
    
    Supports:
    - Winograd Schema Challenge (JSON format)
    - Custom JSON benchmarks (basic support)
    """
    
    def __init__(self, benchmark_dir: Union[str, Path] = "benchmarks"):
        """
        Initialize benchmark suite.
        
        Args:
            benchmark_dir: Directory containing benchmark datasets
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_dummy_winograd()
        self._ensure_dummy_squad()
        self._ensure_dummy_glue()
    
    def _ensure_dummy_winograd(self, filename: str = "winograd_dummy.json"):
        """Create a tiny Winograd-like dataset if none exists."""
        path = self.benchmark_dir / filename
        if path.exists():
            return
        samples = [
            {
                "id": "wsc_0001",
                "text": "The trophy doesn't fit into the brown suitcase because it is too large.",
                "question": "What is too large?",
                "options": ["the trophy", "the suitcase"],
                "answer": "the trophy",
                "difficulty": "medium",
                "reasoning": "coreference"
            },
            {
                "id": "wsc_0002",
                "text": "The trophy doesn't fit into the brown suitcase because it is too small.",
                "question": "What is too small?",
                "options": ["the trophy", "the suitcase"],
                "answer": "the suitcase",
                "difficulty": "medium",
                "reasoning": "coreference"
            },
            {
                "id": "wsc_0003",
                "text": "The city councilmen refused the demonstrators a permit because they feared violence.",
                "question": "Who feared violence?",
                "options": ["the city councilmen", "the demonstrators"],
                "answer": "the city councilmen",
                "difficulty": "hard",
                "reasoning": "commonsense"
            },
            {
                "id": "wsc_0004",
                "text": "The city councilmen refused the demonstrators a permit because they advocated violence.",
                "question": "Who advocated violence?",
                "options": ["the city councilmen", "the demonstrators"],
                "answer": "the demonstrators",
                "difficulty": "hard",
                "reasoning": "commonsense"
            }
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    def _ensure_dummy_squad(self, filename: str = "squad_dummy.json"):
        """Create a tiny SQuAD-like dataset if none exists."""
        path = self.benchmark_dir / filename
        if path.exists():
            return
        samples = [
            {
                "id": "squad_0001",
                "context": "Paris is the capital of France.",
                "question": "What is the capital of France?",
                "answers": ["Paris"]
            },
            {
                "id": "squad_0002",
                "context": "The Pacific Ocean is larger than the Atlantic Ocean.",
                "question": "Which ocean is larger than the Atlantic Ocean?",
                "answers": ["the Pacific Ocean", "Pacific Ocean", "Pacific"]
            }
        ]
        with path.open("w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    def _ensure_dummy_glue(self):
        """Create tiny GLUE-like datasets for supported tasks if none exist."""
        # SST-2
        p = self.benchmark_dir / "sst-2.json"
        if not p.exists():
            s = [
                {"id": "sst2_0001", "sentence": "This movie was fantastic!", "label": "positive"},
                {"id": "sst2_0002", "sentence": "The plot was dull and boring.", "label": "negative"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        # CoLA
        p = self.benchmark_dir / "cola.json"
        if not p.exists():
            s = [
                {"id": "cola_0001", "sentence": "The book is on the table.", "label": "acceptable"},
                {"id": "cola_0002", "sentence": "The book is arrived.", "label": "unacceptable"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        # MRPC
        p = self.benchmark_dir / "mrpc.json"
        if not p.exists():
            s = [
                {"id": "mrpc_0001", "sentence1": "A man is playing guitar.", "sentence2": "A person plays a guitar.", "label": "paraphrase"},
                {"id": "mrpc_0002", "sentence1": "A woman is cooking dinner.", "sentence2": "A man is riding a bike.", "label": "not paraphrase"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        # QQP
        p = self.benchmark_dir / "qqp.json"
        if not p.exists():
            s = [
                {"id": "qqp_0001", "sentence1": "How do I learn Python?", "sentence2": "What is the best way to learn Python?", "label": "paraphrase"},
                {"id": "qqp_0002", "sentence1": "How to bake a cake?", "sentence2": "Where is the nearest gas station?", "label": "not paraphrase"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        # RTE
        p = self.benchmark_dir / "rte.json"
        if not p.exists():
            s = [
                {"id": "rte_0001", "premise": "All cats have tails.", "hypothesis": "Mittens has a tail.", "label": "entailment"},
                {"id": "rte_0002", "premise": "No birds can swim.", "hypothesis": "Some birds can swim.", "label": "not entailment"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        # QNLI (simplified as NLI)
        p = self.benchmark_dir / "qnli.json"
        if not p.exists():
            s = [
                {"id": "qnli_0001", "premise": "The Eiffel Tower is in Paris.", "hypothesis": "Is the Eiffel Tower in Paris?", "label": "entailment"},
                {"id": "qnli_0002", "premise": "Mount Everest is in Nepal.", "hypothesis": "Is Mount Everest in Africa?", "label": "not entailment"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
        # MNLI
        p = self.benchmark_dir / "mnli.json"
        if not p.exists():
            s = [
                {"id": "mnli_0001", "premise": "A boy is running.", "hypothesis": "A child is moving quickly.", "label": "entailment"},
                {"id": "mnli_0002", "premise": "A woman is cooking.", "hypothesis": "No one is in the kitchen.", "label": "contradiction"},
                {"id": "mnli_0003", "premise": "People are sitting in a park.", "hypothesis": "Some people are outside.", "label": "neutral"},
            ]
            with p.open("w", encoding="utf-8") as f:
                json.dump(s, f, ensure_ascii=False, indent=2)
    
    def _load_json_dataset(self, name: str) -> List[Dict]:
        path = self.benchmark_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def run_winograd_benchmark(self, model_path: Union[str, Path], 
                              epsilon_values: List[float],
                              dataset_name: str = "winograd_dummy",
                              seed: Optional[int] = None,
                              schemas: Optional[List[Dict]] = None,
                              device: Optional[str] = None) -> Dict:
        """
        Run Winograd Schema Challenge benchmark.
        
        Args:
            model_path: HF model name or local path
            epsilon_values: List of epsilon values to test
            dataset_name: Dataset JSON name (without extension)
            seed: Random seed for reproducible epsilon masking
            
        Returns:
            Dict with results by epsilon and summary
        """
        if schemas is None:
            schemas = self._load_json_dataset(dataset_name)
        evaluator = WinogradEvaluator(model_name=str(model_path), device=device)

        results_by_epsilon: Dict[str, Dict] = {}
        for eps in epsilon_values:
            detailed = evaluator.evaluate_all_schemas(schemas, epsilon=eps, seed=seed)
            metrics = evaluator.get_performance_metrics(detailed)
            results_by_epsilon[str(eps)] = {
                "metrics": metrics,
                "detailed": detailed,
            }

        # Summary
        best_eps = None
        best_acc = -1.0
        for eps_str, res in results_by_epsilon.items():
            acc = res["metrics"].get("accuracy", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_eps = float(eps_str)

        summary = {
            "best_epsilon": best_eps,
            "best_accuracy": best_acc,
            "num_samples": len(schemas)
        }

        return {
            "benchmark": "winograd",
            "dataset": dataset_name,
            "model": str(model_path),
            "epsilon_values": epsilon_values,
            "results_by_epsilon": results_by_epsilon,
            "summary": summary,
            "created_at": int(time.time())
        }
    
    def run_squad_benchmark(self, model_path: Union[str, Path], 
                           epsilon_values: List[float],
                           dataset_name: str = "squad_dummy",
                           seed: Optional[int] = None,
                           max_new_tokens: int = 32,
                           examples: Optional[List[Dict]] = None,
                           device: Optional[str] = None) -> Dict:
        """
        Run SQuAD-style benchmark using generative answers.

        Args:
            model_path: HF model name or local path
            epsilon_values: List of epsilon values to test
            dataset_name: Dataset JSON name (without extension)
            seed: Random seed for reproducible epsilon masking
            max_new_tokens: Max tokens for generated answers

        Returns:
            Dict with results by epsilon and summary
        """
        try:
            if examples is None:
                exs = self._load_json_dataset(dataset_name)
                chosen_dataset = dataset_name
            else:
                exs = examples
                chosen_dataset = dataset_name
        except FileNotFoundError:
            # Fallback to auto-created dummy dataset
            self._ensure_dummy_squad()
            exs = self._load_json_dataset("squad_dummy")
            chosen_dataset = "squad_dummy"
        evaluator = SquadEvaluator(model_name=str(model_path), device=device)

        results_by_epsilon: Dict[str, Dict] = {}
        for eps in epsilon_values:
            detailed = evaluator.evaluate_all(exs, epsilon=eps, seed=seed, max_new_tokens=max_new_tokens)
            metrics = evaluator.get_performance_metrics(detailed)
            results_by_epsilon[str(eps)] = {
                "metrics": metrics,
                "detailed": detailed,
            }

        # Summary (best F1)
        best_eps = None
        best_f1 = -1.0
        for eps_str, res in results_by_epsilon.items():
            f1 = res["metrics"].get("f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_eps = float(eps_str)

        summary = {
            "best_epsilon": best_eps,
            "best_f1": best_f1,
            "num_samples": len(exs)
        }

        return {
            "benchmark": "squad",
            "dataset": chosen_dataset,
            "model": str(model_path),
            "epsilon_values": epsilon_values,
            "results_by_epsilon": results_by_epsilon,
            "summary": summary,
            "created_at": int(time.time())
        }
    
    def run_glue_benchmark(self, model_path: Union[str, Path], 
                          tasks: List[str],
                          dataset_names: Optional[Dict[str, str]] = None,
                          epsilon_values: Optional[List[float]] = None,
                          seed: Optional[int] = None,
                          device: Optional[str] = None,
                          hf_examples: Optional[Dict[str, List[Dict]]] = None) -> Dict:
        """
        Run GLUE-style benchmarks for selected tasks.

        Args:
            model_path: HF model name or local path
            tasks: List of GLUE task names (e.g., ["SST-2","MRPC"]) 
            dataset_names: Optional mapping task->dataset JSON stem
            epsilon_values: Optional list of epsilon values; defaults to [0.0]
            seed: Optional base seed for masking

        Returns:
            Dict with per-task results and macro summary
        """
        epsilon_values = epsilon_values or [0.0]
        evaluator = GlueEvaluator(model_name=str(model_path), device=device)

        per_task_results: Dict[str, Dict] = {}
        for task in tasks:
            ds_name = dataset_names.get(task, task.lower()) if dataset_names else task.lower()
            try:
                if hf_examples and task in hf_examples:
                    examples = hf_examples[task]
                    chosen_ds = ds_name
                else:
                    examples = self._load_json_dataset(ds_name)
                    chosen_ds = ds_name
            except FileNotFoundError:
                # Fallback to default dummy for this task
                self._ensure_dummy_glue()
                default_ds = task.lower()
                examples = self._load_json_dataset(default_ds)
                chosen_ds = default_ds

            results_by_epsilon: Dict[str, Dict] = {}
            for eps in epsilon_values:
                detailed = evaluator.evaluate_all(task, examples, epsilon=eps, seed=seed)
                metrics = evaluator.get_performance_metrics(task, detailed)
                results_by_epsilon[str(eps)] = {
                    "metrics": metrics,
                    "detailed": detailed,
                }

            # pick best epsilon by accuracy
            best_eps = None
            best_acc = -1.0
            for eps_str, res in results_by_epsilon.items():
                acc = res["metrics"].get("accuracy", 0.0)
                if acc > best_acc:
                    best_acc = acc
                    best_eps = float(eps_str)

            per_task_results[task] = {
                "dataset": chosen_ds,
                "results_by_epsilon": results_by_epsilon,
                "summary": {
                    "best_epsilon": best_eps,
                    "best_accuracy": best_acc,
                    "num_samples": len(examples)
                }
            }

        # Macro summary (average best accuracies)
        task_accs = [info["summary"]["best_accuracy"] for info in per_task_results.values()]
        macro_avg_acc = float(sum(task_accs) / len(task_accs)) if task_accs else 0.0

        return {
            "benchmark": "glue",
            "model": str(model_path),
            "tasks": tasks,
            "epsilon_values": epsilon_values,
            "per_task": per_task_results,
            "summary": {
                "macro_avg_accuracy": macro_avg_acc,
                "num_tasks": len(tasks)
            },
            "created_at": int(time.time())
        }
    
    def run_custom_benchmark(self, model_path: Union[str, Path], 
                           benchmark_config: Dict) -> Dict:
        """Placeholder for future custom benchmark support."""
        raise NotImplementedError("Custom benchmark is not implemented yet.")
    
    def run_full_evaluation(self, model_path: Union[str, Path], 
                          epsilon_values: List[float],
                          dataset_name: str = "winograd_dummy",
                          seed: Optional[int] = None) -> Dict:
        """
        Run full evaluation suite (currently Winograd only).
        
        Args:
            model_path: HF model name or local path
            epsilon_values: List of epsilon values to test
            dataset_name: Dataset JSON name (without extension)
            seed: Random seed for reproducible epsilon masking
            
        Returns:
            Dict with evaluation results
        """
        winograd = self.run_winograd_benchmark(model_path, epsilon_values, dataset_name, seed)
        return {"winograd": winograd}
    
    def save_benchmark_results(self, results: Dict, 
                             output_path: Union[str, Path]):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def load_benchmark_results(self, input_path: Union[str, Path]) -> Dict:
        input_path = Path(input_path)
        with input_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_available_benchmarks(self) -> List[str]:
        return [p.stem for p in self.benchmark_dir.glob("*.json")]
    
    def get_benchmark_info(self, benchmark_name: str) -> Dict:
        path = self.benchmark_dir / f"{benchmark_name}.json"
        if not path.exists():
            return {"exists": False, "path": str(path)}
        data = self._load_json_dataset(benchmark_name)
        return {
            "exists": True,
            "path": str(path),
            "num_samples": len(data),
            "sample_keys": list(data[0].keys()) if data else []
        }
