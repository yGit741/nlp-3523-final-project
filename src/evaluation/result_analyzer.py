"""
Result analysis and performance evaluation.

This module provides comprehensive analysis of evaluation results
including statistical analysis and performance insights.
"""

from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class ResultAnalyzer:
    """
    Analyzes evaluation results and performance metrics.
    
    Features:
    - Statistical analysis
    - Performance comparison
    - Trend analysis
    - Significance testing
    """
    
    def __init__(self):
        """Initialize result analyzer."""
        # No heavy initialization needed
        pass
    
    def analyze_performance_trends(self, results_by_epsilon: Dict[float, Dict]) -> Dict:
        """
        Analyze performance trends across epsilon values.
        
        Args:
            results_by_epsilon: Results organized by epsilon value
            
        Returns:
            Performance trend analysis
        """
        # Accept mapping from epsilon-> {metrics: {...}}
        if "results_by_epsilon" in results_by_epsilon:
            mapping = results_by_epsilon["results_by_epsilon"]
        else:
            mapping = results_by_epsilon

        eps = sorted([float(e) for e in mapping.keys()])
        # Choose primary metric key (prefer accuracy, then f1, then em)
        sample_metrics = None
        for v in mapping.values():
            if isinstance(v, dict) and "metrics" in v:
                sample_metrics = v["metrics"]
                break
        metric_key = None
        if sample_metrics:
            for k in ("accuracy", "f1", "em"):
                if k in sample_metrics:
                    metric_key = k
                    break
        metric_key = metric_key or (list(sample_metrics.keys())[0] if sample_metrics else "accuracy")

        vals = np.array([
            (mapping[str(e)]["metrics"].get(metric_key) if str(e) in mapping else mapping[e]["metrics"].get(metric_key))
            for e in eps
        ])

        # Simple trend: correlation between epsilon and the chosen metric
        corr = float(np.corrcoef(eps, vals)[0, 1]) if len(eps) > 1 else 0.0
        best_idx = int(np.argmax(vals)) if len(vals) > 0 else -1
        best_eps = eps[best_idx] if best_idx >= 0 else None
        best_val = float(vals[best_idx]) if best_idx >= 0 else 0.0

        return {
            "metric": metric_key,
            "epsilon_values": eps,
            "values": vals.tolist(),
            "correlation_epsilon_metric": corr,
            "best_epsilon": best_eps,
            "best_value": best_val,
        }
    
    def compare_model_performance(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Compare performance across different models.
        
        Args:
            model_results: Results for different models
            
        Returns:
            Model comparison analysis
        """
        rows = []
        for name, res in model_results.items():
            if "results_by_epsilon" in res:
                rb = res["results_by_epsilon"]
            elif "winograd" in res:
                rb = res["winograd"]["results_by_epsilon"]
            else:
                rb = res
            best_acc = 0.0
            best_eps = None
            for eps, data in rb.items():
                acc = data["metrics"]["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_eps = float(eps)
            rows.append({"model": name, "best_accuracy": best_acc, "best_epsilon": best_eps})
        df = pd.DataFrame(rows).sort_values("best_accuracy", ascending=False)
        return {"leaderboard": df.to_dict(orient="records")}
    
    def calculate_statistical_significance(self, results1: List[float], 
                                         results2: List[float]) -> Dict:
        """
        Calculate statistical significance between two result sets.
        
        Args:
            results1: First set of results
            results2: Second set of results
            
        Returns:
            Statistical significance analysis
        """
        if len(results1) < 2 or len(results2) < 2:
            return {"t_test_p": None, "mannwhitney_p": None, "cohen_d": None}
        t_stat, t_p = stats.ttest_ind(results1, results2, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(results1, results2, alternative="two-sided")
        # Cohen's d
        m1, m2 = np.mean(results1), np.mean(results2)
        s1, s2 = np.std(results1, ddof=1), np.std(results2, ddof=1)
        # Pooled SD
        n1, n2 = len(results1), len(results2)
        sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 0.0
        d = (m1 - m2) / sp if sp > 0 else None
        return {"t_test_p": float(t_p), "mannwhitney_p": float(u_p), "cohen_d": float(d) if d is not None else None}
    
    def analyze_performance_distribution(self, results: List[Dict]) -> Dict:
        """
        Analyze performance distribution across samples.
        
        Args:
            results: Evaluation results
            
        Returns:
            Performance distribution analysis
        """
        confs = [r.get("confidence", 0.0) for r in results]
        accs = [1.0 if r.get("is_correct") else 0.0 for r in results]
        return {
            "confidence_mean": float(np.mean(confs)) if confs else 0.0,
            "confidence_std": float(np.std(confs)) if confs else 0.0,
            "accuracy_mean": float(np.mean(accs)) if accs else 0.0,
            "accuracy_std": float(np.std(accs)) if accs else 0.0,
        }
    
    def identify_performance_factors(self, results: List[Dict]) -> Dict:
        """
        Identify factors affecting performance.
        
        Args:
            results: Evaluation results
            
        Returns:
            Performance factor analysis
        """
        df = pd.DataFrame(results)
        if df.empty:
            return {}
        df["difficulty"].fillna("unknown", inplace=True)
        df["is_correct_int"] = df["is_correct"].astype(int)
        by_diff = df.groupby("difficulty")["is_correct_int"].mean().to_dict()
        # Approximate context length by original text length
        df["context_len"] = df["original_text"].astype(str).apply(len)
        corr = float(np.corrcoef(df["context_len"], df["is_correct_int"])[0, 1]) if len(df) > 1 else 0.0
        return {"accuracy_by_difficulty": by_diff, "corr_contextlen_accuracy": corr}
    
    def calculate_confidence_intervals(self, results: List[float], 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for performance metrics.
        
        Args:
            results: Performance results
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        arr = np.array(results)
        if arr.size == 0:
            return (0.0, 0.0)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        z = 1.96 if abs(confidence_level - 0.95) < 1e-6 else stats.norm.ppf((1 + confidence_level) / 2.0)
        half_width = z * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
        return (mean - half_width, mean + half_width)
    
    def analyze_correlation_patterns(self, results: List[Dict]) -> Dict:
        """
        Analyze correlation patterns in results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Correlation analysis
        """
        df = pd.DataFrame(results)
        if df.empty:
            return {"correlations": {}}
        df["is_correct_int"] = df["is_correct"].astype(int)
        corr = {}
        if "confidence" in df:
            corr["confidence_accuracy"] = float(np.corrcoef(df["confidence"], df["is_correct_int"])[0, 1]) if len(df) > 1 else 0.0
        if "original_text" in df:
            df["context_len"] = df["original_text"].astype(str).apply(len)
            corr["contextlen_accuracy"] = float(np.corrcoef(df["context_len"], df["is_correct_int"])[0, 1]) if len(df) > 1 else 0.0
        return {"correlations": corr}
    
    def generate_performance_summary(self, results: Dict) -> Dict:
        """
        Generate performance summary statistics.
        
        Args:
            results: Evaluation results
            
        Returns:
            Performance summary
        """
        trend = self.analyze_performance_trends(results)
        return {
            "metric": trend.get("metric"),
            "best_epsilon": trend.get("best_epsilon"),
            "best_value": trend.get("best_value"),
            "correlation_epsilon_metric": trend.get("correlation_epsilon_metric"),
        }
    
    def create_performance_report(self, results: Dict, 
                                model_name: str) -> Dict:
        """
        Create comprehensive performance report.
        
        Args:
            results: Evaluation results
            model_name: Model name
            
        Returns:
            Comprehensive performance report
        """
        summary = self.generate_performance_summary(results)
        return {
            "model": model_name,
            "summary": summary,
        }
    
    def export_analysis_results(self, analysis: Dict, 
                              output_path: Union[str, Path]):
        """
        Export analysis results to file.
        
        Args:
            analysis: Analysis results
            output_path: Output file path
        """
        import json
        from pathlib import Path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    def load_analysis_results(self, input_path: Union[str, Path]) -> Dict:
        """
        Load analysis results from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Loaded analysis results
        """
        import json
        from pathlib import Path
        p = Path(input_path)
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def validate_results(self, results: Dict) -> Tuple[bool, List[str]]:
        """
        Validate evaluation results.
        
        Args:
            results: Results to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors: List[str] = []
        if not isinstance(results, dict):
            return False, ["Results must be a dict"]
        if "results_by_epsilon" not in results and "winograd" not in results:
            errors.append("Missing 'results_by_epsilon' or 'winograd' key")
        return (len(errors) == 0, errors)
    
    def analyze_epsilon_performance(self, results: Dict) -> Dict:
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