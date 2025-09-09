"""
Evaluation CLI for Knowledge vs Reasoning Separation project.

Usage:
    python -m src.evaluation.evaluate --model gpt2 \
        --epsilons 0.0,0.1,0.3 --benchmark winograd_dummy \
        --output results/eval_results.json
"""

from .benchmark_suite import BenchmarkSuite


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.0,0.1,0.3",
        help="Comma-separated epsilon values (e.g., 0.0,0.1,0.3)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="winograd_dummy",
        help="Benchmark dataset name (without .json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output JSON path",
    )

    args = parser.parse_args()
    epsilon_values = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]

    # Ensure benchmarks directory under this package (creates dummy if missing)
    suite = BenchmarkSuite(benchmark_dir=Path(__file__).parent / "benchmarks")

    # Currently only Winograd is wired into full evaluation
    results = suite.run_full_evaluation(args.model, epsilon_values, dataset_name=args.benchmark)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()


