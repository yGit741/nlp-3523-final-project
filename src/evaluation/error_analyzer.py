"""
Detailed error analysis for model evaluation.

This module provides comprehensive error analysis capabilities
for understanding model failures and performance patterns.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict, Counter


class ErrorAnalyzer:
    """
    Analyzes errors in model evaluation results.
    
    Features:
    - Error categorization
    - Pattern identification
    - Statistical analysis
    - Detailed reporting
    """
    
    def __init__(self):
        """Initialize error analyzer."""
        # No heavy init required for basic analysis
        self._supported_reason_keys = {"coreference", "commonsense", "syntax", "logic", "unknown"}
    
    def analyze_winograd_errors(self, results: List[Dict]) -> Dict:
        """
        Analyze errors in Winograd Schema evaluation.
        
        Args:
            results: Winograd evaluation results
            
        Returns:
            Detailed error analysis
        """
        errors = [r for r in results if not r.get("is_correct")]
        correct = [r for r in results if r.get("is_correct")]

        # Counts
        total = len(results)
        error_count = len(errors)
        correct_count = len(correct)

        # By difficulty
        errors_by_difficulty: Dict[str, int] = defaultdict(int)
        for e in errors:
            errors_by_difficulty[e.get("difficulty", "unknown")] += 1

        # By reasoning type
        errors_by_reasoning: Dict[str, int] = defaultdict(int)
        for e in errors:
            key = e.get("reasoning", "unknown")
            errors_by_reasoning[key] += 1

        # Confidence stats
        avg_conf_correct = float(np.mean([r.get("confidence", 0.0) for r in correct])) if correct else 0.0
        avg_conf_errors = float(np.mean([r.get("confidence", 0.0) for r in errors])) if errors else 0.0

        # Most confusing examples (highest confidence errors)
        top_confident_errors = sorted(errors, key=lambda x: x.get("confidence", 0.0), reverse=True)[:10]

        return {
            "total": total,
            "error_count": error_count,
            "correct_count": correct_count,
            "errors_by_difficulty": dict(errors_by_difficulty),
            "errors_by_reasoning": dict(errors_by_reasoning),
            "avg_confidence_correct": avg_conf_correct,
            "avg_confidence_errors": avg_conf_errors,
            "top_confident_errors": top_confident_errors,
        }
    
    def categorize_errors(self, results: List[Dict]) -> Dict:
        """
        Categorize errors by type and pattern.
        
        Args:
            results: Evaluation results
            
        Returns:
            Error categorization results
        """
        errors = [r for r in results if not r.get("is_correct")]
        categories: Dict[str, int] = defaultdict(int)
        for e in errors:
            reason = e.get("reasoning", "unknown")
            categories[reason] += 1
        return dict(categories)
    
    def analyze_confidence_patterns(self, results: List[Dict]) -> Dict:
        """
        Analyze confidence patterns in predictions.
        
        Args:
            results: Evaluation results
            
        Returns:
            Confidence analysis results
        """
        df = pd.DataFrame(results)
        if df.empty:
            return {"avg_confidence": 0.0, "calibration": []}
        df["is_correct_int"] = df["is_correct"].astype(int)
        avg_conf = float(df["confidence"].mean()) if "confidence" in df else 0.0
        # Simple calibration: bin confidence and compute accuracy per bin
        bins = np.linspace(0.0, 1.0, 11)
        df["conf_bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)
        cal = df.groupby("conf_bin").agg(acc=("is_correct_int", "mean"), count=("is_correct_int", "size")).reset_index()
        calibration = [
            {
                "bin": str(b),
                "midpoint": float(b.mid),
                "accuracy": float(a),
                "count": int(c),
            }
            for b, a, c in zip(cal["conf_bin"], cal["acc"], cal["count"])
        ]
        return {"avg_confidence": avg_conf, "calibration": calibration}
    
    def identify_error_patterns(self, results: List[Dict]) -> Dict:
        """
        Identify common error patterns.
        
        Args:
            results: Evaluation results
            
        Returns:
            Error pattern analysis
        """
        # Simple pattern surfacing by question phrasing and option length/lexicality
        errors = [r for r in results if not r.get("is_correct")]
        question_starts = Counter([r.get("question", "").split(" ")[0].lower() for r in errors])
        option_lengths = [len((r.get("predicted_answer") or "").split()) for r in errors]
        return {
            "common_question_starts": dict(question_starts.most_common(10)),
            "avg_error_option_length": float(np.mean(option_lengths)) if option_lengths else 0.0,
        }
    
    def analyze_difficulty_errors(self, results: List[Dict]) -> Dict:
        """
        Analyze errors by difficulty level.
        
        Args:
            results: Evaluation results
            
        Returns:
            Difficulty-based error analysis
        """
        df = pd.DataFrame(results)
        if df.empty:
            return {}
        df["difficulty"].fillna("unknown", inplace=True)
        df["is_error"] = ~df["is_correct"]
        agg = df.groupby("difficulty").agg(
            error_rate=("is_error", "mean"),
            count=("is_error", "size"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()
        return agg.to_dict(orient="list")
    
    def compare_error_patterns(self, results_by_epsilon: Dict[float, List[Dict]]) -> Dict:
        """
        Compare error patterns across epsilon values.
        
        Args:
            results_by_epsilon: Results organized by epsilon value
            
        Returns:
            Cross-epsilon error comparison
        """
        per_eps = {}
        for eps, res in results_by_epsilon.items():
            per_eps[float(eps)] = self.categorize_errors(res)
        return per_eps
    
    def generate_error_report(self, results: List[Dict], 
                            model_name: str, epsilon: float) -> Dict:
        """
        Generate comprehensive error report.
        
        Args:
            results: Evaluation results
            model_name: Model name
            epsilon: Epsilon value
            
        Returns:
            Comprehensive error report
        """
        summary = self.analyze_winograd_errors(results)
        patterns = self.identify_error_patterns(results)
        confidence = self.analyze_confidence_patterns(results)
        return {
            "model": model_name,
            "epsilon": epsilon,
            "summary": summary,
            "patterns": patterns,
            "confidence": confidence,
        }
    
    def create_error_visualizations(self, results: List[Dict], 
                                  output_dir: Union[str, Path]):
        """
        Create error analysis visualizations.
        
        Args:
            results: Evaluation results
            output_dir: Output directory for visualizations
        """
        from .visualizer import ResultVisualizer
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        viz = ResultVisualizer()
        # Difficulty analysis
        try:
            viz.plot_difficulty_analysis(results, output_path=output_dir / "difficulty.png")
        except Exception:
            pass
        # Confidence analysis
        try:
            viz.plot_confidence_analysis(results, output_path=output_dir / "confidence.png")
        except Exception:
            pass
    
    def export_error_data(self, results: List[Dict], 
                         output_path: Union[str, Path]):
        """
        Export error data for further analysis.
        
        Args:
            results: Evaluation results
            output_path: Output file path
        """
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def get_error_statistics(self, results: List[Dict]) -> Dict:
        """
        Get statistical summary of errors.
        
        Args:
            results: Evaluation results
            
        Returns:
            Error statistics dictionary
        """
        total = len(results)
        errors = [r for r in results if not r.get("is_correct")]
        error_rate = (len(errors) / total) if total > 0 else 0.0
        confs = [r.get("confidence", 0.0) for r in results]
        return {
            "total": total,
            "errors": len(errors),
            "error_rate": error_rate,
            "avg_confidence": float(np.mean(confs)) if confs else 0.0,
        }
    
    def identify_problematic_samples(self, results: List[Dict], 
                                   threshold: float = 0.5) -> List[Dict]:
        """
        Identify problematic samples for further analysis.
        
        Args:
            results: Evaluation results
            threshold: Confidence threshold for problematic samples
            
        Returns:
            List of problematic samples
        """
        problems = []
        for r in results:
            conf = r.get("confidence", 0.0)
            if (not r.get("is_correct") and conf >= threshold) or (r.get("is_correct") and conf <= (1 - threshold)):
                problems.append(r)
        return problems
