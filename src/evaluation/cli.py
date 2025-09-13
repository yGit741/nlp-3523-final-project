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

  - All:
      python -m src.evaluation.cli --suite all --model gpt2 \
        --epsilons 0.0,0.1 --seed 42 --output results/all.json
"""

from .benchmark_suite import BenchmarkSuite
from .visualizer import ResultVisualizer
from .result_analyzer import ResultAnalyzer
from .error_analyzer import ErrorAnalyzer


def main():
    import argparse
    import json
    from pathlib import Path
    import time

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
        "--output_dir",
        type=str,
        default="",
        help="If set, write a structured run folder here; otherwise auto-create under results/",
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

    # Create structured run directory <model>_<benchmark>_<timestamp>
    model_label = str(args.model).replace("/", "-")
    if args.suite == "winograd":
        bench_label = winograd_ds
    elif args.suite == "squad":
        bench_label = squad_ds
    elif args.suite == "glue":
        bench_label = "glue"
    else:
        bench_label = "all"
    timestamp = int(time.time())
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path("results") / f"{model_label}_{bench_label}_{timestamp}"
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save combined results.json
    with (run_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    viz = ResultVisualizer()
    ra = ResultAnalyzer()
    ea = ErrorAnalyzer()

    # Generate suite-specific artifacts
    if args.suite == "winograd":
        # Save suite-level JSON
        with (run_dir / "winograd_results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # Per-epsilon detailed results
        for eps_key, data in results.get("results_by_epsilon", {}).items():
            detailed = data.get("detailed", [])
            with (run_dir / f"winograd_detailed_eps_{eps_key}.json").open("w", encoding="utf-8") as f:
                json.dump(detailed, f, ensure_ascii=False, indent=2)
        # Plots
        viz.plot_epsilon_performance(results, output_path=plots_dir / "winograd_epsilon.png")
        # Error/difficulty/confidence for best epsilon
        best_eps = results.get("summary", {}).get("best_epsilon")
        if best_eps is not None:
            key = str(best_eps)
            if key in results.get("results_by_epsilon", {}):
                best_detailed = results["results_by_epsilon"][key]["detailed"]
                try:
                    viz.plot_difficulty_analysis(best_detailed, output_path=plots_dir / "winograd_difficulty.png")
                except Exception:
                    pass
                try:
                    viz.plot_confidence_analysis(best_detailed, output_path=plots_dir / "winograd_confidence.png")
                except Exception:
                    pass
                # Error report
                try:
                    err_report = ea.generate_error_report(best_detailed, model_name=args.model, epsilon=float(best_eps))
                    with (reports_dir / "winograd_error_report.json").open("w", encoding="utf-8") as f:
                        json.dump(err_report, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        # Performance report
        try:
            perf_report = ra.create_performance_report(results, model_name=args.model)
            with (reports_dir / "winograd_performance_report.json").open("w", encoding="utf-8") as f:
                json.dump(perf_report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    elif args.suite == "squad":
        # Save suite-level JSON
        with (run_dir / "squad_results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # Per-epsilon detailed
        for eps_key, data in results.get("results_by_epsilon", {}).items():
            detailed = data.get("detailed", [])
            with (run_dir / f"squad_detailed_eps_{eps_key}.json").open("w", encoding="utf-8") as f:
                json.dump(detailed, f, ensure_ascii=False, indent=2)
        # Epsilon performance plot (metric-agnostic: will use F1 if present)
        viz.plot_epsilon_performance(results, output_path=plots_dir / "squad_epsilon.png")
        # Performance report
        try:
            perf_report = ra.create_performance_report(results, model_name=args.model)
            with (reports_dir / "squad_performance_report.json").open("w", encoding="utf-8") as f:
                json.dump(perf_report, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    elif args.suite == "glue":
        # Save suite-level JSON
        with (run_dir / "glue_results.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # Per-task artifacts
        per_task = results.get("per_task", {}) or {}
        for task, task_data in per_task.items():
            rb = task_data.get("results_by_epsilon", {})
            # Per-epsilon detailed saves
            for eps_key, data in rb.items():
                detailed = data.get("detailed", [])
                with (run_dir / f"glue_{task}_detailed_eps_{eps_key}.json").open("w", encoding="utf-8") as f:
                    json.dump(detailed, f, ensure_ascii=False, indent=2)
            # Epsilon plot per task
            try:
                viz.plot_epsilon_performance(rb, output_path=plots_dir / f"glue_{task}_epsilon.png")
            except Exception:
                pass
            # Performance report per task
            try:
                trend = ra.analyze_performance_trends(rb)
                with (reports_dir / f"glue_{task}_performance_report.json").open("w", encoding="utf-8") as f:
                    json.dump({"task": task, "trend": trend}, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        # Macro summary
        try:
            with (reports_dir / "glue_macro_summary.json").open("w", encoding="utf-8") as f:
                json.dump(results.get("summary", {}), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    elif args.suite == "all":
        # Save combined components
        if results.get("winograd"):
            with (run_dir / "winograd_results.json").open("w", encoding="utf-8") as f:
                json.dump(results["winograd"], f, ensure_ascii=False, indent=2)
            viz.plot_epsilon_performance(results["winograd"], output_path=plots_dir / "winograd_epsilon.png")
            best_eps = results["winograd"].get("summary", {}).get("best_epsilon")
            if best_eps is not None:
                key = str(best_eps)
                if key in results["winograd"].get("results_by_epsilon", {}):
                    best_detailed = results["winograd"]["results_by_epsilon"][key]["detailed"]
                    try:
                        viz.plot_difficulty_analysis(best_detailed, output_path=plots_dir / "winograd_difficulty.png")
                    except Exception:
                        pass
                    try:
                        viz.plot_confidence_analysis(best_detailed, output_path=plots_dir / "winograd_confidence.png")
                    except Exception:
                        pass
                    try:
                        err_report = ea.generate_error_report(best_detailed, model_name=args.model, epsilon=float(best_eps))
                        with (reports_dir / "winograd_error_report.json").open("w", encoding="utf-8") as f:
                            json.dump(err_report, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
            try:
                perf_report = ra.create_performance_report(results["winograd"], model_name=args.model)
                with (reports_dir / "winograd_performance_report.json").open("w", encoding="utf-8") as f:
                    json.dump(perf_report, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if results.get("squad"):
            with (run_dir / "squad_results.json").open("w", encoding="utf-8") as f:
                json.dump(results["squad"], f, ensure_ascii=False, indent=2)
            viz.plot_epsilon_performance(results["squad"], output_path=plots_dir / "squad_epsilon.png")
            try:
                perf_report = ra.create_performance_report(results["squad"], model_name=args.model)
                with (reports_dir / "squad_performance_report.json").open("w", encoding="utf-8") as f:
                    json.dump(perf_report, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        if results.get("glue"):
            with (run_dir / "glue_results.json").open("w", encoding="utf-8") as f:
                json.dump(results["glue"], f, ensure_ascii=False, indent=2)
            per_task = results["glue"].get("per_task", {}) or {}
            for task, task_data in per_task.items():
                rb = task_data.get("results_by_epsilon", {})
                # plots
                try:
                    viz.plot_epsilon_performance(rb, output_path=plots_dir / f"glue_{task}_epsilon.png")
                except Exception:
                    pass
                # per-epsilon details
                for eps_key, data in rb.items():
                    detailed = data.get("detailed", [])
                    with (run_dir / f"glue_{task}_detailed_eps_{eps_key}.json").open("w", encoding="utf-8") as f:
                        json.dump(detailed, f, ensure_ascii=False, indent=2)
                # report
                try:
                    trend = ra.analyze_performance_trends(rb)
                    with (reports_dir / f"glue_{task}_performance_report.json").open("w", encoding="utf-8") as f:
                        json.dump({"task": task, "trend": trend}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            try:
                with (reports_dir / "glue_macro_summary.json").open("w", encoding="utf-8") as f:
                    json.dump(results["glue"].get("summary", {}), f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    # Also write to the legacy flat output path if provided
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved structured outputs to {run_dir}\nLegacy results JSON saved to {out_path}")


if __name__ == "__main__":
    main()


