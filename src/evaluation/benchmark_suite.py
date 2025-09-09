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
    
    def _load_json_dataset(self, name: str) -> List[Dict]:
        path = self.benchmark_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def run_winograd_benchmark(self, model_path: Union[str, Path], 
                              epsilon_values: List[float],
                              dataset_name: str = "winograd_dummy",
                              seed: Optional[int] = None) -> Dict:
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
        schemas = self._load_json_dataset(dataset_name)
        evaluator = WinogradEvaluator(model_name=str(model_path))

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
                           epsilon_values: List[float]) -> Dict:
        """Placeholder for future SQuAD support."""
        raise NotImplementedError("SQuAD benchmark is not implemented yet.")
    
    def run_glue_benchmark(self, model_path: Union[str, Path], 
                          tasks: List[str]) -> Dict:
        """Placeholder for future GLUE support."""
        raise NotImplementedError("GLUE benchmark is not implemented yet.")
    
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
    
    def compare_epsilon_performance(self, results: Dict) -> Dict:
        """
        Compare performance across epsilon values for Winograd results.
        """
        eps_to_acc = {float(eps): data["metrics"]["accuracy"] for eps, data in results.get("results_by_epsilon", {}).items()}
        best_eps = max(eps_to_acc, key=eps_to_acc.get) if eps_to_acc else None
        return {
            "accuracy_by_epsilon": eps_to_acc,
            "best_epsilon": best_eps,
            "best_accuracy": eps_to_acc.get(best_eps, 0.0) if best_eps is not None else 0.0
        }
    
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
