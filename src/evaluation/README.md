# Evaluation CLI

Command-line interface to run Winograd, SQuAD, and GLUE benchmarks.

## Datasets directory
- Benchmarks are read from `src/evaluation/benchmarks/`.
- Auto-created dummy files (for quick runs):
  - Winograd: `winograd_dummy.json`
  - SQuAD: `squad_dummy.json`
  - GLUE: `sst-2.json`, `cola.json`, `mrpc.json`, `qqp.json`, `rte.json`, `qnli.json`, `mnli.json`
- You can place your own datasets in the same folder and reference by name (without `.json`).

## Basic usage (run from project root)

### Winograd
```bash
python -m src.evaluation.cli --suite winograd --model gpt2 \
  --epsilons 0.0,0.1,0.3 --winograd_benchmark winograd_dummy \
  --seed 42 --output results/winograd.json
```

### SQuAD
```bash
python -m src.evaluation.cli --suite squad --model gpt2 \
  --epsilons 0.0,0.2 --squad_benchmark squad_dummy \
  --seed 42 --max_new_tokens 32 --output results/squad.json
```

### GLUE
```bash
python -m src.evaluation.cli --suite glue --model gpt2 \
  --glue_tasks SST-2,MRPC --glue_datasets SST-2:sst-2,MRPC:mrpc \
  --epsilons 0.0,0.1 --seed 42 --output results/glue.json
```

### Run all suites
```bash
python -m src.evaluation.cli --suite all --model gpt2 --epsilons 0.0,0.1 \
  --winograd_benchmark winograd_dummy --squad_benchmark squad_dummy \
  --glue_tasks SST-2,MRPC --glue_datasets SST-2:sst-2,MRPC:mrpc \
  --seed 42 --max_new_tokens 32 --output results/all.json
```

## Arguments
- `--suite`: one of `winograd`, `squad`, `glue`, `all`.
- `--model`: HF model name or local path (e.g., `gpt2`).
- `--epsilons`: comma-separated epsilon values for masking (e.g., `0.0,0.1`).
- `--output`: path to save JSON results.
- `--benchmark`: generic dataset name (used if task-specific not provided).
- `--winograd_benchmark`: dataset for Winograd (defaults to `--benchmark`).
- `--squad_benchmark`: dataset for SQuAD (defaults to `--benchmark`).
- `--seed`: random seed for deterministic masking (optional).
- `--max_new_tokens`: max tokens for SQuAD generation (default: 32).
- GLUE-specific:
  - `--glue_tasks`: comma-separated tasks (e.g., `SST-2,MRPC`).
  - `--glue_datasets`: optional task:name mapping (e.g., `SST-2:sst-2,MRPC:mrpc`).

## Outputs
- Single-suite runs return a JSON object with metrics and detailed results.
- `--suite all` returns a JSON object with `winograd`, `squad`, and `glue` sections.

## Notes
- If a requested dataset JSON is missing, Winograd/SQuAD/GLUE runners fall back to dummy datasets created automatically.
