"""
Evaluation CLI for Knowledge vs Reasoning Separation project.

Usage examples:
  - Winograd (direct from Hugging Face, default: WinoGrande m validation):
      python -m src.evaluation.cli --suite winograd --model gpt2 \
        --epsilons 0.0,0.1 --seed 42 \
        --output results/winogrande_m_val.json

    Customize the HF dataset/config/split (e.g., xs validation):
      python -m src.evaluation.cli --suite winograd --model gpt2 \
        --epsilons 0.0 --hf_winograd winogrande:xs:validation --seed 42 \
        --output results/winogrande_xs_val.json

    Use a local JSON dataset instead of HF (disable HF by passing an empty string):
      python -m src.evaluation.cli --suite winograd --model gpt2 \
        --epsilons 0.0,0.1 --hf_winograd "" \
        --winograd_benchmark winograd_dummy \
        --benchmark_dir path/to/benchmarks --output results/winograd_local.json

  - SQuAD (direct from Hugging Face; default: squad:validation):
      python -m src.evaluation.cli --suite squad --model gpt2 \
        --epsilons 0.0,0.2 --seed 42 --max_new_tokens 32 \
        --output results/squad_val.json

    Customize dataset/split or use local JSON:
      # Different split via HF
      python -m src.evaluation.cli --suite squad --model gpt2 \
        --hf_squad squad:validation --epsilons 0.0 --seed 42 \
        --max_new_tokens 32 --output results/squad_val.json

      # Disable HF and use local JSON
      python -m src.evaluation.cli --suite squad --model gpt2 \
        --hf_squad "" --squad_benchmark squad_dummy --epsilons 0.0 \
        --max_new_tokens 32 --output results/squad_local.json

  - GLUE (direct from Hugging Face; default tasks: SST-2, MRPC on validation):
      python -m src.evaluation.cli --suite glue --model gpt2 \
        --epsilons 0.0,0.1 --seed 42 --output results/glue.json

    Customize tasks/splits or use local JSON:
      # Different splits via HF
      python -m src.evaluation.cli --suite glue --model gpt2 \
        --hf_glue SST-2:validation,MRPC:validation --epsilons 0.0 \
        --seed 42 --output results/glue_val.json

      # Disable HF and use local JSON mappings
      python -m src.evaluation.cli --suite glue --model gpt2 \
        --hf_glue "" --glue_tasks SST-2,MRPC \
        --glue_datasets SST-2:sst-2,MRPC:mrpc --epsilons 0.0 \
        --seed 42 --output results/glue_local.json

  - All:
      # Uses HF defaults: WinoGrande (m, val), SQuAD (validation), GLUE (SST-2 & MRPC, validation)
      python -m src.evaluation.cli --suite all --model gpt2 \
        --epsilons 0.0,0.1 --seed 42 --max_new_tokens 32 \
        --output results/all.json

Notes:
- --benchmark_dir optionally points to a custom datasets folder for local JSONs.
- HF loaders: --hf_winograd <dataset:config:split>, --hf_squad <dataset:split>, --hf_glue <TASK:split,...>.
- Pass an empty string (e.g., --hf_squad "") to disable HF for that suite and use local JSONs instead.
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
        "--benchmark_dir",
        type=str,
        default="",
        help="Optional path to datasets directory (overrides default benchmarks folder)",
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
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cuda', 'mps', 'cpu', or 'auto' (prefers CUDA, then MPS).",
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
    # Direct-from-HF for SQuAD and GLUE
    parser.add_argument(
        "--hf_squad",
        type=str,
        default="squad:validation",
        help="If set, format is '<dataset>:<split>' e.g., 'squad:validation'. Bypasses local JSON.",
    )
    # Direct-from-HF for Winograd (WinoGrande)
    parser.add_argument(
        "--hf_winograd",
        type=str,
        default="winogrande:xs:validation",
        help="If set, format is '<dataset>:<config>:<split>' e.g., 'winogrande:m:validation'. Bypasses local JSON.",
    )
    parser.add_argument(
        "--hf_glue",
        type=str,
        default="SST-2:validation,MRPC:validation",
        help="If set, comma-separated '<TASK>:<split>' entries (e.g., 'SST-2:validation,MRPC:validation'). Bypasses local JSON.",
    )
    parser.add_argument(
        "--glue_datasets",
        type=str,
        default="",
        help="Optional task:name mapping, e.g., SST-2:sst-2,MRPC:mrpc",
    )

    args = parser.parse_args()
    epsilon_values = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]

    # Ensure benchmarks directory
    default_dir = Path(__file__).parent / "benchmarks"
    bench_dir = Path(args.benchmark_dir) if args.benchmark_dir else default_dir
    suite = BenchmarkSuite(benchmark_dir=bench_dir)

    def _resolve_device(user_choice: str) -> str:
        import torch
        if user_choice and user_choice.lower() != "auto":
            return user_choice
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    resolved_device = _resolve_device(args.device)

    # Dispatch by suite
    if args.suite == "winograd":
        # Optional HF path for WinoGrande
        schemas_mem = None
        winograd_ds = args.winograd_benchmark or args.benchmark
        if args.hf_winograd:
            try:
                from datasets import load_dataset
                spec = args.hf_winograd
                parts = spec.split(":")
                if len(parts) != 3:
                    raise SystemExit("--hf_winograd must be '<dataset>:<config>:<split>' e.g., 'winogrande:m:validation'")
                ds_name, ds_config, ds_split = parts
                # Normalize short configs like 'm' -> 'winogrande_m'
                short_sizes = {"xs", "s", "m", "l", "xl", "debiased"}
                ds_config_norm = f"winogrande_{ds_config}" if (ds_config in short_sizes and not ds_config.startswith("winogrande_")) else ds_config
                hf_ds = load_dataset(ds_name, ds_config_norm, split=ds_split)
                schemas_mem = []
                for ex in hf_ds:
                    # WinoGrande mapping
                    options = [ex.get("option1"), ex.get("option2")]
                    ans_index = int(ex.get("answer")) - 1 if isinstance(ex.get("answer"), (int, str)) else None
                    answer_text = options[ans_index] if ans_index in (0, 1) else None
                    schemas_mem.append({
                        "id": ex.get("idx"),
                        "text": ex.get("sentence"),
                        "question": "Which option best completes the sentence?",
                        "options": options,
                        "answer": answer_text,
                        "difficulty": None,
                        "reasoning": "commonsense",
                    })
                winograd_ds = f"{ds_name}_{ds_config_norm}_{ds_split}"
            except ImportError:
                raise SystemExit("The 'datasets' package is required for --hf_winograd. Please install it.")
        results = suite.run_winograd_benchmark(args.model, epsilon_values, dataset_name=winograd_ds, seed=args.seed, schemas=schemas_mem, device=resolved_device)
    elif args.suite == "squad":
        # Optional HF path for SQuAD
        squad_ds = args.squad_benchmark or args.benchmark
        examples_mem = None
        if args.hf_squad:
            try:
                from datasets import load_dataset
                spec = args.hf_squad
                parts = spec.split(":")
                if len(parts) != 2:
                    raise SystemExit("--hf_squad must be '<dataset>:<split>' e.g., 'squad:validation'")
                ds_name, ds_split = parts
                hf_ds = load_dataset(ds_name, split=ds_split)
                examples_mem = []
                for ex in hf_ds:
                    examples_mem.append({
                        "id": ex.get("id"),
                        "context": ex.get("context"),
                        "question": ex.get("question"),
                        "answers": (ex.get("answers", {}) or {}).get("text", []),
                    })
                squad_ds = f"{ds_name}_{ds_split}"
            except ImportError:
                raise SystemExit("The 'datasets' package is required for --hf_squad. Please install it.")
        results = suite.run_squad_benchmark(args.model, epsilon_values, dataset_name=squad_ds, seed=args.seed, max_new_tokens=args.max_new_tokens, examples=examples_mem, device=resolved_device)
    elif args.suite == "glue":
        # Optional HF for GLUE
        tasks = [t.strip() for t in args.glue_tasks.split(",") if t.strip()]
        hf_examples: dict = {}
        if args.hf_glue:
            try:
                from datasets import load_dataset
                specs = [s.strip() for s in args.hf_glue.split(",") if s.strip()]
                for spec in specs:
                    if ":" not in spec:
                        raise SystemExit("Each --hf_glue entry must be '<TASK>:<split>'")
                    task_name, split_name = spec.split(":", 1)
                    cfg = task_name.lower().replace("-", "")
                    hf_ds = load_dataset("glue", cfg, split=split_name)
                    rows = []
                    for ex in hf_ds:
                        if task_name in ("SST-2","CoLA"):
                            # label names may exist in features
                            names = getattr(hf_ds.features["label"], "names", None)
                            if task_name == "SST-2":
                                label = names[ex["label"]] if names else ("positive" if ex["label"] == 1 else "negative")
                                rows.append({"id": ex.get("idx"), "sentence": ex.get("sentence"), "label": label})
                            else:
                                label = names[ex["label"]] if names else ("acceptable" if ex["label"] == 1 else "unacceptable")
                                rows.append({"id": ex.get("idx"), "sentence": ex.get("sentence"), "label": label})
                        elif task_name in ("MRPC","QQP"):
                            names = getattr(hf_ds.features["label"], "names", None)
                            label = names[ex["label"]] if names else ("paraphrase" if ex["label"] == 1 else "not paraphrase")
                            s1 = ex.get("sentence1") or ex.get("question1")
                            s2 = ex.get("sentence2") or ex.get("question2")
                            rows.append({"id": ex.get("idx"), "sentence1": s1, "sentence2": s2, "label": label})
                        elif task_name in ("RTE","QNLI"):
                            names = getattr(hf_ds.features["label"], "names", None)
                            raw = names[ex["label"]] if names else ("entailment" if ex["label"] == 0 else "not_entailment")
                            label = raw.replace("_", " ")
                            prem = ex.get("sentence1") or ex.get("sentence")
                            hyp = ex.get("sentence2") or ex.get("question")
                            rows.append({"id": ex.get("idx"), "premise": prem, "hypothesis": hyp, "label": label})
                        elif task_name == "MNLI":
                            names = getattr(hf_ds.features["label"], "names", None)
                            label = names[ex["label"]] if names else ["entailment","neutral","contradiction"][ex["label"]]
                            rows.append({"id": ex.get("idx"), "premise": ex.get("premise"), "hypothesis": ex.get("hypothesis"), "label": label})
                        else:
                            raise SystemExit(f"Unsupported GLUE task in --hf_glue: {task_name}")
                    hf_examples[task_name] = rows
                # If HF provided but --glue_tasks empty, infer from specs
                if not tasks:
                    tasks = list(hf_examples.keys())
            except ImportError:
                raise SystemExit("The 'datasets' package is required for --hf_glue. Please install it.")
        if not tasks:
            raise SystemExit("--glue_tasks is required when --suite glue (or provide --hf_glue)")
        mapping: dict = {}
        if args.glue_datasets:
            for pair in args.glue_datasets.split(","):
                if not pair.strip():
                    continue
                if ":" not in pair:
                    raise SystemExit(f"Invalid --glue_datasets entry: {pair}")
                task_name, ds_name = pair.split(":", 1)
                mapping[task_name.strip()] = ds_name.strip()
        results = suite.run_glue_benchmark(args.model, tasks, dataset_names=mapping or None, epsilon_values=epsilon_values, seed=args.seed, device=resolved_device, hf_examples=hf_examples or None)
    elif args.suite == "all":
        # Winograd
        winograd_ds = args.winograd_benchmark or args.benchmark
        schemas_mem = None
        if args.hf_winograd:
            try:
                from datasets import load_dataset
                spec = args.hf_winograd
                parts = spec.split(":")
                if len(parts) != 3:
                    raise SystemExit("--hf_winograd must be '<dataset>:<config>:<split>' e.g., 'winogrande:m:validation'")
                ds_name, ds_config, ds_split = parts
                short_sizes = {"xs", "s", "m", "l", "xl", "debiased"}
                ds_config_norm = f"winogrande_{ds_config}" if (ds_config in short_sizes and not ds_config.startswith("winogrande_")) else ds_config
                hf_ds = load_dataset(ds_name, ds_config_norm, split=ds_split)
                schemas_mem = []
                for ex in hf_ds:
                    options = [ex.get("option1"), ex.get("option2")]
                    ans_index = int(ex.get("answer")) - 1 if isinstance(ex.get("answer"), (int, str)) else None
                    answer_text = options[ans_index] if ans_index in (0, 1) else None
                    schemas_mem.append({
                        "id": ex.get("idx"),
                        "text": ex.get("sentence"),
                        "question": "Which option best completes the sentence?",
                        "options": options,
                        "answer": answer_text,
                        "difficulty": None,
                        "reasoning": "commonsense",
                    })
                winograd_ds = f"{ds_name}_{ds_config_norm}_{ds_split}"
            except ImportError:
                raise SystemExit("The 'datasets' package is required for --hf_winograd. Please install it.")
        winograd_res = suite.run_winograd_benchmark(args.model, epsilon_values, dataset_name=winograd_ds, seed=args.seed, schemas=schemas_mem, device=resolved_device)

        # SQuAD
        squad_ds = args.squad_benchmark or args.benchmark
        examples_mem = None
        if args.hf_squad:
            try:
                from datasets import load_dataset
                spec = args.hf_squad
                parts = spec.split(":")
                if len(parts) != 2:
                    raise SystemExit("--hf_squad must be '<dataset>:<split>' e.g., 'squad:validation'")
                ds_name, ds_split = parts
                hf_ds = load_dataset(ds_name, split=ds_split)
                examples_mem = []
                for ex in hf_ds:
                    examples_mem.append({
                        "id": ex.get("id"),
                        "context": ex.get("context"),
                        "question": ex.get("question"),
                        "answers": (ex.get("answers", {}) or {}).get("text", []),
                    })
                squad_ds = f"{ds_name}_{ds_split}"
            except ImportError:
                raise SystemExit("The 'datasets' package is required for --hf_squad. Please install it.")
        squad_res = suite.run_squad_benchmark(args.model, epsilon_values, dataset_name=squad_ds, seed=args.seed, max_new_tokens=args.max_new_tokens, examples=examples_mem, device=resolved_device)

        # GLUE
        tasks = [t.strip() for t in args.glue_tasks.split(",") if t.strip()]
        hf_examples: dict = {}
        if args.hf_glue:
            try:
                from datasets import load_dataset
                specs = [s.strip() for s in args.hf_glue.split(",") if s.strip()]
                for spec in specs:
                    if ":" not in spec:
                        raise SystemExit("Each --hf_glue entry must be '<TASK>:<split>'")
                    task_name, split_name = spec.split(":", 1)
                    cfg = task_name.lower().replace("-", "")
                    hf_ds = load_dataset("glue", cfg, split=split_name)
                    rows = []
                    for ex in hf_ds:
                        if task_name in ("SST-2","CoLA"):
                            names = getattr(hf_ds.features["label"], "names", None)
                            if task_name == "SST-2":
                                label = names[ex["label"]] if names else ("positive" if ex["label"] == 1 else "negative")
                                rows.append({"id": ex.get("idx"), "sentence": ex.get("sentence"), "label": label})
                            else:
                                label = names[ex["label"]] if names else ("acceptable" if ex["label"] == 1 else "unacceptable")
                                rows.append({"id": ex.get("idx"), "sentence": ex.get("sentence"), "label": label})
                        elif task_name in ("MRPC","QQP"):
                            names = getattr(hf_ds.features["label"], "names", None)
                            label = names[ex["label"]] if names else ("paraphrase" if ex["label"] == 1 else "not paraphrase")
                            s1 = ex.get("sentence1") or ex.get("question1")
                            s2 = ex.get("sentence2") or ex.get("question2")
                            rows.append({"id": ex.get("idx"), "sentence1": s1, "sentence2": s2, "label": label})
                        elif task_name in ("RTE","QNLI"):
                            names = getattr(hf_ds.features["label"], "names", None)
                            raw = names[ex["label"]] if names else ("entailment" if ex["label"] == 0 else "not_entailment")
                            label = raw.replace("_", " ")
                            prem = ex.get("sentence1") or ex.get("sentence")
                            hyp = ex.get("sentence2") or ex.get("question")
                            rows.append({"id": ex.get("idx"), "premise": prem, "hypothesis": hyp, "label": label})
                        elif task_name == "MNLI":
                            names = getattr(hf_ds.features["label"], "names", None)
                            label = names[ex["label"]] if names else ["entailment","neutral","contradiction"][ex["label"]]
                            rows.append({"id": ex.get("idx"), "premise": ex.get("premise"), "hypothesis": ex.get("hypothesis"), "label": label})
                        else:
                            raise SystemExit(f"Unsupported GLUE task in --hf_glue: {task_name}")
                    hf_examples[task_name] = rows
                if not tasks:
                    tasks = list(hf_examples.keys())
            except ImportError:
                raise SystemExit("The 'datasets' package is required for --hf_glue. Please install it.")
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
            glue_res = suite.run_glue_benchmark(args.model, tasks, dataset_names=mapping or None, epsilon_values=epsilon_values, seed=args.seed, device=resolved_device, hf_examples=hf_examples or None)

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


