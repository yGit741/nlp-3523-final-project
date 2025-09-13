"""
Result visualization for evaluation analysis.

This module provides comprehensive visualization capabilities
for evaluation results and analysis.
"""

from typing import List, Dict, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


class ResultVisualizer:
    """
    Creates visualizations for evaluation results.
    
    Features:
    - Performance plots
    - Error analysis charts
    - Statistical visualizations
    - Interactive plots
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (12, 8)):
        """
        Initialize result visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        try:
            plt.style.use(self.style)
        except Exception:
            # Fallback to default style if unavailable
            plt.style.use("default")
        sns.set_context("talk")
    
    def plot_epsilon_performance(self, results_by_epsilon: Dict[float, Dict], 
                               output_path: Optional[Union[str, Path]] = None):
        """
        Plot performance vs epsilon values.
        
        Args:
            results_by_epsilon: Results organized by epsilon
            output_path: Optional output path for saving plot
        """
        # Accept either direct mapping {eps: {metrics...}} or full suite output
        if "results_by_epsilon" in results_by_epsilon:
            mapping = results_by_epsilon["results_by_epsilon"]
        else:
            mapping = results_by_epsilon

        eps_values = sorted([float(e) for e in mapping.keys()])
        # Determine primary metric to plot
        first_metrics = None
        for v in mapping.values():
            if isinstance(v, dict) and "metrics" in v:
                first_metrics = v["metrics"]
                break
        metric_key = None
        if first_metrics:
            for k in ("accuracy", "f1", "em"):
                if k in first_metrics:
                    metric_key = k
                    break
        metric_key = metric_key or (list(first_metrics.keys())[0] if first_metrics else "accuracy")

        # Collect y values for the chosen metric
        metric_values = [
            (mapping[str(e)]["metrics"].get(metric_key) if str(e) in mapping else mapping[e]["metrics"].get(metric_key))
            for e in eps_values
        ]

        # Avg confidence (if available)
        avg_conf = [
            (mapping[str(e)]["metrics"].get("avg_confidence", 0.0) if str(e) in mapping else mapping[e]["metrics"].get("avg_confidence", 0.0))
            for e in eps_values
        ]

        # Error counts if available; fall back to EM-derived errors
        error_counts = []
        for e in eps_values:
            m = mapping[str(e)]["metrics"] if str(e) in mapping else mapping[e]["metrics"]
            if "total" in m and "correct" in m:
                errors = m["total"] - m["correct"]
            elif "total" in m and "em" in m:
                # Approximate errors from EM proportion
                errors = int(round(m["total"] * (1.0 - float(m["em"]))))
            else:
                errors = 0
            error_counts.append(errors)

        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        axes[0].plot(eps_values, metric_values, marker="o")
        axes[0].set_title("Performance vs Epsilon")
        axes[0].set_xlabel("Epsilon")
        ylabel = {"accuracy": "Accuracy", "f1": "F1", "em": "EM"}.get(metric_key, metric_key)
        axes[0].set_ylabel(ylabel)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(eps_values, avg_conf, marker="s", color="#2a9d8f")
        axes[1].set_title("Avg Confidence vs Epsilon")
        axes[1].set_xlabel("Epsilon")
        axes[1].set_ylabel("Avg Confidence")
        axes[1].grid(True, alpha=0.3)

        axes[2].bar([str(e) for e in eps_values], error_counts, color="#e76f51")
        axes[2].set_title("Error Count vs Epsilon")
        axes[2].set_xlabel("Epsilon")
        axes[2].set_ylabel("Errors")
        axes[2].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, axes
    
    def plot_difficulty_analysis(self, results: List[Dict], 
                               output_path: Optional[Union[str, Path]] = None):
        """
        Plot performance analysis by difficulty.
        
        Args:
            results: Evaluation results
            output_path: Optional output path for saving plot
        """
        # Aggregate by difficulty
        df = pd.DataFrame(results)
        if df.empty:
            raise ValueError("No results provided for difficulty analysis")
        df["difficulty"].fillna("unknown", inplace=True)
        agg = df.groupby("difficulty").agg(
            total=("is_correct", "count"),
            correct=("is_correct", "sum"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()
        agg["accuracy"] = agg["correct"] / agg["total"].replace(0, np.nan)

        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        sns.barplot(ax=axes[0], data=agg, x="difficulty", y="accuracy", color="#457b9d")
        axes[0].set_title("Accuracy by Difficulty")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, axis="y", alpha=0.3)

        sns.barplot(ax=axes[1], data=agg, x="difficulty", y="total", color="#e9c46a")
        axes[1].set_title("Count by Difficulty")
        axes[1].grid(True, axis="y", alpha=0.3)

        sns.barplot(ax=axes[2], data=agg, x="difficulty", y="avg_confidence", color="#2a9d8f")
        axes[2].set_title("Avg Confidence by Difficulty")
        axes[2].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, axes
    
    def plot_confidence_analysis(self, results: List[Dict], 
                               output_path: Optional[Union[str, Path]] = None):
        """
        Plot confidence analysis.
        
        Args:
            results: Evaluation results
            output_path: Optional output path for saving plot
        """
        df = pd.DataFrame(results)
        if df.empty:
            raise ValueError("No results provided for confidence analysis")
        df["is_correct_int"] = df["is_correct"].astype(int)

        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Distribution by correctness
        sns.kdeplot(ax=axes[0], data=df, x="confidence", hue="is_correct", common_norm=False, fill=True)
        axes[0].set_title("Confidence Distribution (Correct vs Error)")
        axes[0].set_xlim(0, 1)
        axes[0].grid(True, alpha=0.3)

        # Scatter confidence vs correctness (jittered)
        jitter = (np.random.rand(len(df)) - 0.5) * 0.05
        axes[1].scatter(df["confidence"], df["is_correct_int"] + jitter, alpha=0.5)
        axes[1].set_title("Confidence vs Correctness")
        axes[1].set_xlabel("Confidence")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Error", "Correct"])
        axes[1].set_xlim(0, 1)
        axes[1].grid(True, alpha=0.3)

        # Calibration bins
        bins = np.linspace(0.0, 1.0, 11)
        df["conf_bin"] = pd.cut(df["confidence"], bins=bins, include_lowest=True)
        cal = df.groupby("conf_bin").agg(acc=("is_correct_int", "mean"), count=("is_correct_int", "size")).reset_index()
        axes[2].plot([b.mid for b in cal["conf_bin"].cat.categories], cal["acc"], marker="o")
        axes[2].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[2].set_title("Calibration (Accuracy vs Confidence)")
        axes[2].set_xlabel("Confidence bin midpoint")
        axes[2].set_ylabel("Empirical Accuracy")
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, axes
    
    def plot_error_patterns(self, error_analysis: Dict, 
                          output_path: Optional[Union[str, Path]] = None):
        """
        Plot error pattern analysis.
        
        Args:
            error_analysis: Error analysis results
            output_path: Optional output path for saving plot
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        by_reason = error_analysis.get("errors_by_reasoning", {})
        if by_reason:
            reasons = list(by_reason.keys())
            counts = [by_reason[r] for r in reasons]
            sns.barplot(ax=axes[0], x=reasons, y=counts, color="#e76f51")
            axes[0].set_title("Errors by Reasoning Type")
            axes[0].tick_params(axis='x', rotation=30)
            axes[0].grid(True, axis="y", alpha=0.3)
        else:
            axes[0].axis('off')

        by_diff = error_analysis.get("errors_by_difficulty", {})
        if by_diff:
            diffs = list(by_diff.keys())
            counts = [by_diff[d] for d in diffs]
            sns.barplot(ax=axes[1], x=diffs, y=counts, color="#457b9d")
            axes[1].set_title("Errors by Difficulty")
            axes[1].grid(True, axis="y", alpha=0.3)
        else:
            axes[1].axis('off')

        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, axes
    
    def plot_model_comparison(self, model_results: Dict[str, Dict], 
                            output_path: Optional[Union[str, Path]] = None):
        """
        Plot model comparison.
        
        Args:
            model_results: Results for different models
            output_path: Optional output path for saving plot
        """
        names = []
        best_accs = []
        for name, res in model_results.items():
            # Accept either full suite or winograd block
            if "results_by_epsilon" in res:
                rb = res["results_by_epsilon"]
            elif "winograd" in res:
                rb = res["winograd"]["results_by_epsilon"]
            else:
                rb = res
            # Best accuracy across eps
            accs = []
            for eps, data in rb.items():
                accs.append(data["metrics"]["accuracy"])
            names.append(name)
            best_accs.append(max(accs) if accs else 0.0)

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 6))
        sns.barplot(ax=ax, x=names, y=best_accs, color="#264653")
        ax.set_ylim(0, 1)
        ax.set_title("Best Accuracy by Model")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, ax
    
    def plot_statistical_analysis(self, analysis_results: Dict, 
                                output_path: Optional[Union[str, Path]] = None):
        """
        Plot statistical analysis results.
        
        Args:
            analysis_results: Statistical analysis results
            output_path: Optional output path for saving plot
        """
        # This is a generic placeholder that attempts to visualize provided arrays
        fig, ax = plt.subplots(figsize=self.figsize)
        if "distribution" in analysis_results:
            data = analysis_results["distribution"]
            sns.histplot(data, bins=20, kde=True, ax=ax)
            ax.set_title("Distribution")
        elif "correlation_matrix" in analysis_results and "labels" in analysis_results:
            mat = np.array(analysis_results["correlation_matrix"])
            labels = analysis_results["labels"]
            sns.heatmap(mat, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
        else:
            ax.text(0.5, 0.5, "No statistical analysis to plot", ha='center', va='center')
            ax.axis('off')
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, ax
    
    def create_dashboard(self, results: Dict, 
                        output_path: Union[str, Path]):
        """
        Create comprehensive results dashboard.
        
        Args:
            results: Complete evaluation results
            output_path: Output path for dashboard
        """
        # Basic dashboard for Winograd results
        if "winograd" in results:
            winograd = results["winograd"]
        else:
            winograd = results
        fig1, _ = self.plot_epsilon_performance(winograd)
        fig2 = None
        # If possible, plot difficulty analysis for best epsilon
        try:
            rb = winograd["results_by_epsilon"]
            # Select metric for best epsilon
            metric_key = None
            sample_metrics = None
            if rb:
                sample_metrics = next(iter(rb.values())).get("metrics", {})
            if sample_metrics:
                for k in ("accuracy", "f1", "em"):
                    if k in sample_metrics:
                        metric_key = k
                        break
            metric_key = metric_key or "accuracy"
            best_eps = max(rb, key=lambda k: rb[k]["metrics"].get(metric_key, 0.0)) if rb else None
            if best_eps is not None:
                detailed = rb[best_eps]["detailed"]
                fig2, _ = self.plot_difficulty_analysis(detailed)
        except Exception:
            pass
        # Save concatenated dashboard as separate images for simplicity
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig1.savefig(output_path.with_name(output_path.stem + "_epsilon.png"), dpi=300, bbox_inches="tight")
        if fig2 is not None:
            fig2.savefig(output_path.with_name(output_path.stem + "_difficulty.png"), dpi=300, bbox_inches="tight")
        return str(output_path)
    
    def plot_training_curves(self, training_history: Dict, 
                           output_path: Optional[Union[str, Path]] = None):
        """
        Plot training curves.
        
        Args:
            training_history: Training history data
            output_path: Optional output path for saving plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        if "loss" in training_history:
            ax.plot(training_history["loss"], label="loss")
        if "val_loss" in training_history:
            ax.plot(training_history["val_loss"], label="val_loss")
        if "accuracy" in training_history:
            ax.plot(training_history["accuracy"], label="accuracy")
        ax.set_title("Training Curves")
        ax.set_xlabel("Step/Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, ax
    
    def create_heatmap(self, data: np.ndarray, labels: List[str], 
                      title: str, output_path: Optional[Union[str, Path]] = None):
        """
        Create correlation or performance heatmap.
        
        Args:
            data: 2D data array
            labels: Axis labels
            title: Plot title
            output_path: Optional output path for saving plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(data, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(title)
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, ax
    
    def plot_distribution(self, data: List[float], title: str, 
                         output_path: Optional[Union[str, Path]] = None):
        """
        Plot data distribution.
        
        Args:
            data: Data to plot
            title: Plot title
            output_path: Optional output path for saving plot
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        sns.histplot(data, bins=20, kde=True, ax=axes[0])
        axes[0].set_title(f"Histogram - {title}")
        sns.violinplot(y=data, ax=axes[1], color="#a8dadc")
        axes[1].set_title(f"Violin - {title}")
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return fig, axes
    
    def save_plot(self, fig, output_path: Union[str, Path], 
                 format: str = "png", dpi: int = 300):
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            output_path: Output file path
            format: File format (png, pdf, svg)
            dpi: Resolution for raster formats
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.with_suffix(f".{format}"), format=format, dpi=dpi, bbox_inches="tight")
    
    def set_plot_style(self, style: str):
        """
        Set plotting style.
        
        Args:
            style: Matplotlib style name
        """
        self.style = style
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")
        sns.set_context("talk")
