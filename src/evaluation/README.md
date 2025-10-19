# Evaluation CLI

Command-line interface to run Winograd, SQuAD, and GLUE benchmarks.

## Datasets directory
- Benchmarks are read from `src/evaluation/benchmarks/` by default, or from a custom folder via `--benchmark_dir`.
- Auto-created dummy files (for quick runs):
  - Winograd: `winograd_dummy.json`
  - SQuAD: `squad_dummy.json`
  - GLUE: `sst-2.json`, `cola.json`, `mrpc.json`, `qqp.json`, `rte.json`, `qnli.json`, `mnli.json`
- You can place your own datasets in the same folder and reference by name (without `.json`).
  - For Winograd, you can also load directly from Hugging Face Datasets with `--hf_winograd`.

## Basic usage (run from project root)

### Winograd (direct from Hugging Face; default)
```bash
python -m src.evaluation.cli --suite winograd --model gpt2 \
  --epsilons 0.0,0.1 --seed 42 \
  --output results/winogrande_m_val.json
```

Customize dataset/config/split (e.g., xs validation):
```bash
python -m src.evaluation.cli --suite winograd --model gpt2 \
  --epsilons 0.0 --hf_winograd winogrande:xs:validation --seed 42 \
  --output results/winogrande_xs_val.json
```

Use a local JSON dataset instead of HF (disable HF by passing empty string):
```bash
python -m src.evaluation.cli --suite winograd --model gpt2 \
  --epsilons 0.0,0.1 --hf_winograd "" \
  --winograd_benchmark winograd_dummy \
  --benchmark_dir src/evaluation/benchmarks \
  --seed 42 --output results/winograd_local.json
```

### SQuAD (direct from Hugging Face; default)
```bash
python -m src.evaluation.cli --suite squad --model gpt2 \
  --epsilons 0.0,0.2 --seed 42 --max_new_tokens 32 \
  --output results/squad_val.json
```

Customize dataset/split (e.g., SQuAD v1.1 validation is default):
```bash
python -m src.evaluation.cli --suite squad --model gpt2 \
  --hf_squad squad:validation --epsilons 0.0 --seed 42 \
  --output results/squad_val.json
```

Use a local JSON dataset (disable HF by passing empty string):
```bash
python -m src.evaluation.cli --suite squad --model gpt2 \
  --hf_squad "" --squad_benchmark squad_dummy \
  --benchmark_dir src/evaluation/benchmarks --epsilons 0.0 \
  --seed 42 --max_new_tokens 32 --output results/squad_local.json
```

### GLUE (direct from Hugging Face; default example)
```bash
python -m src.evaluation.cli --suite glue --model gpt2 \
  --glue_tasks SST-2,MRPC --epsilons 0.0,0.1 --seed 42 \
  --output results/glue.json
```

Customize via HF (task:split pairs):
```bash
python -m src.evaluation.cli --suite glue --model gpt2 \
  --hf_glue SST-2:validation,MRPC:validation --epsilons 0.0 \
  --seed 42 --output results/glue_val.json
```

Use local JSON datasets instead (disable HF):
```bash
python -m src.evaluation.cli --suite glue --model gpt2 \
  --hf_glue "" --glue_tasks SST-2,MRPC \
  --glue_datasets SST-2:sst-2,MRPC:mrpc \
  --benchmark_dir src/evaluation/benchmarks --epsilons 0.0 \
  --seed 42 --output results/glue_local.json
```

### Run all suites
```bash
python -m src.evaluation.cli --suite all --model gpt2 --epsilons 0.0,0.1 \
  --seed 42 --max_new_tokens 32 --output results/all.json
```

## Arguments
- `--suite`: one of `winograd`, `squad`, `glue`, `all`.
- `--model`: HF model name or local path (e.g., `gpt2`).
- `--epsilons`: comma-separated epsilon values for masking (e.g., `0.0,0.1`).
- `--output`: path to save JSON results.
- `--benchmark_dir`: optional path to datasets directory (for local JSON datasets).
- `--benchmark`: generic dataset name (used if task-specific not provided).
- `--winograd_benchmark`: dataset for Winograd (defaults to `--benchmark`).
- `--squad_benchmark`: dataset for SQuAD (defaults to `--benchmark`).
- `--seed`: random seed for deterministic masking (optional).
- `--max_new_tokens`: max tokens for SQuAD generation (default: 32).
- SQuAD-specific:
  - `--hf_squad`: direct HF loading in the form `<dataset>:<split>` (e.g., `squad:validation`).
    - Set to an empty string `""` to disable HF and use local JSON via `--squad_benchmark`.
- Winograd-specific:
  - `--hf_winograd`: direct HF loading in the form `<dataset>:<config>:<split>` (e.g., `winogrande:m:validation`).
    - Set to an empty string `""` to disable HF and use local JSON via `--winograd_benchmark`.
- GLUE-specific:
  - `--glue_tasks`: comma-separated tasks (e.g., `SST-2,MRPC`).
  - `--glue_datasets`: optional task:name mapping (e.g., `SST-2:sst-2,MRPC:mrpc`).
  - `--hf_glue`: comma-separated `<TASK>:<split>` entries to load directly from HF (e.g., `SST-2:validation,MRPC:validation`).
    - Set to an empty string `""` to disable HF and use local JSON via `--glue_datasets`.

## Outputs
- Single-suite runs return a JSON object with metrics and detailed results.
- `--suite all` returns a JSON object with `winograd`, `squad`, and `glue` sections.

## Notes
- Winograd defaults to loading WinoGrande (m, validation) via Hugging Face Datasets.
- SQuAD defaults to loading `squad:validation` via Hugging Face Datasets.
- GLUE can be loaded directly from HF via `--hf_glue` (otherwise provide local JSONs).
- To use local JSON for Winograd, pass `--hf_winograd ""` and provide `--winograd_benchmark` (and optionally `--benchmark_dir`).
- SQuAD and GLUE continue to read local JSONs; if a requested dataset JSON is missing, they fall back to dummy datasets created automatically.
