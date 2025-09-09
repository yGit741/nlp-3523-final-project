"""
Evaluation CLI for Knowledge vs Reasoning Separation project.

Usage examples:
  - Winograd:
      python -m src.evaluation.cli --suite winograd --model gpt2 \
        --epsilons 0.0,0.1,0.3 --benchmark winograd_dummy \
        --output results/winograd.json

  - SQuAD:
      python -m src.evaluation.cli --suite squad --model gpt2 \
        --epsilons 0.0,0.2 --benchmark squad_dev --seed 42 \
        --max_new_tokens 32 --output results/squad.json

  - GLUE:
      python -m src.evaluation.cli --suite glue --model gpt2 \
        --glue_tasks SST-2,MRPC --glue_datasets SST-2:sst-2,MRPC:mrpc \
        --epsilons 0.0,0.1 --seed 42 --output results/glue.json
"""

from .benchmark_suite import BenchmarkSuite


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument(
        "--suite",
        type=str,
        default="winograd",
        choices=["winograd", "squad", "glue", "all"],
        help="Which evaluation suite to run",
    )
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
        help="Generic dataset name (used if task-specific not provided)",
    )
    parser.add_argument(
        "--winograd_benchmark",
        type=str,
        default=None,
        help="Winograd dataset name (without .json). Defaults to --benchmark",
    )
    parser.add_argument(
        "--squad_benchmark",
        type=str,
        default=None,
        help="SQuAD dataset name (without .json). Defaults to --benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for masking (optional)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max new tokens for SQuAD generation",
    )
    # GLUE-specific
    parser.add_argument(
        "--glue_tasks",
        type=str,
        default="",
        help="Comma-separated GLUE task names (e.g., SST-2,MRPC)",
    )
    parser.add_argument(
        "--glue_datasets",
        type=str,
        default="",
        help="Optional task:name mapping, e.g., SST-2:sst-2,MRPC:mrpc",
    )

    args = parser.parse_args()
    epsilon_values = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]

    # Ensure benchmarks directory under this package (creates dummy if missing)
    suite = BenchmarkSuite(benchmark_dir=Path(__file__).parent / "benchmarks")

    # Dispatch by suite
    if args.suite == "winograd":
        winograd_ds = args.winograd_benchmark or args.benchmark
        results = suite.run_winograd_benchmark(args.model, epsilon_values, dataset_name=winograd_ds, seed=args.seed)
    elif args.suite == "squad":
        squad_ds = args.squad_benchmark or args.benchmark
        results = suite.run_squad_benchmark(args.model, epsilon_values, dataset_name=squad_ds, seed=args.seed, max_new_tokens=args.max_new_tokens)
    elif args.suite == "glue":
        tasks = [t.strip() for t in args.glue_tasks.split(",") if t.strip()]
        if not tasks:
            raise SystemExit("--glue_tasks is required when --suite glue")
        mapping: dict = {}
        if args.glue_datasets:
            for pair in args.glue_datasets.split(","):
                if not pair.strip():
                    continue
                if ":" not in pair:
                    raise SystemExit(f"Invalid --glue_datasets entry: {pair}")
                task_name, ds_name = pair.split(":", 1)
                mapping[task_name.strip()] = ds_name.strip()
        results = suite.run_glue_benchmark(args.model, tasks, dataset_names=mapping or None, epsilon_values=epsilon_values, seed=args.seed)
    elif args.suite == "all":
        # Winograd
        winograd_ds = args.winograd_benchmark or args.benchmark
        winograd_res = suite.run_winograd_benchmark(args.model, epsilon_values, dataset_name=winograd_ds, seed=args.seed)

        # SQuAD
        squad_ds = args.squad_benchmark or args.benchmark
        squad_res = suite.run_squad_benchmark(args.model, epsilon_values, dataset_name=squad_ds, seed=args.seed, max_new_tokens=args.max_new_tokens)

        # GLUE
        tasks = [t.strip() for t in args.glue_tasks.split(",") if t.strip()]
        mapping: dict = {}
        if args.glue_datasets:
            for pair in args.glue_datasets.split(","):
                if not pair.strip():
                    continue
                if ":" not in pair:
                    raise SystemExit(f"Invalid --glue_datasets entry: {pair}")
                task_name, ds_name = pair.split(":", 1)
                mapping[task_name.strip()] = ds_name.strip()
        glue_res = None
        if tasks:
            glue_res = suite.run_glue_benchmark(args.model, tasks, dataset_names=mapping or None, epsilon_values=epsilon_values, seed=args.seed)

        results = {
            "winograd": winograd_res,
            "squad": squad_res,
            "glue": glue_res,
        }
    else:
        raise SystemExit(f"Unknown suite: {args.suite}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved evaluation results to {out_path}")


if __name__ == "__main__":
    main()


