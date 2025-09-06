# Knowledge vs Reasoning Separation: NLP-3523 Final Project

This repository contains a **skeleton implementation** for the final project investigating whether restricting knowledge (ε-masking) can actually help reasoning tasks like Winograd Schema Challenge.

## 🎯 Project Goal

**"Sometimes less knowledge leads to better reasoning"**

This repository's main goal is to **divide the work** among three team members (Gilad, Yonatan, Omer) to implement a complete pipeline for testing whether controlled knowledge masking can improve reasoning performance.

## 🚀 Getting Started

**Start with the demo**: Open `not-to-know-poc.ipynb` to understand the complete research methodology, implementation, and expected results (it doesn't run yet, but a lot of code can be taken from them and it show the whole flow).

## 👥 Team Work Division

### Gilad (Data & Preprocessing)
**Deliverable 1**: `python -m src.data_prep.prepare_data --dataset <name> --output <path>`  
**Deliverable 2**: `data/processed/<dataset_name>/`

**Specific Data Processing Tasks:**
- **Dataset Selection**: Choose and load datasets (WikiText-103, OSCAR, custom)
- **Text Cleaning**: Remove HTML, normalize whitespace, handle encoding
- **Sentence Segmentation**: Split text into sentences with character-level spans
- **Named Entity Recognition (NER)**: Extract entities with spaCy, assign unique IDs
- **Part-of-Speech (POS) Tagging**: Tokenize and tag with POS labels
- **IOB Tagging**: Convert NER to Inside-Outside-Beginning format
- **Punctuation Extraction**: Identify and extract punctuation marks with positions
- **Special Token Detection**: Find URLs, emails, numbers with positions
- **Data Validation**: Ensure all spans are valid and non-overlapping
- **Format Output**: Create JSONL files with exact structure below

**Output Format** (each sample):
```json
{
  "id": "wiki_v1_0000456",
  "text": "Turing—brilliant! See https://turing.org.uk... Really.",
  "sent_spans": [{"start":0,"end":20},{"start":21,"end":49},{"start":50,"end":57}],
  "punct_spans": [
    {"start":6,"end":7,"value":"—"},
    {"start":15,"end":16,"value":"!"},
    {"start":40,"end":43,"value":"..."},
    {"start":56,"end":57,"value":"."}
  ],
  "special_tags": [
    {"start":25,"end":46,"type":"URL","value":"https://turing.org.uk"}
  ],
  "ner_spans": [
    {"entity_id":"e0001","start":0,"end":6,"label":"PERSON"}
  ],
  "pos_tokens": ["Turing—brilliant!", "See", "https://turing.org.uk...", "Really."],
  "pos_tags":   ["PROPN",            "VERB","X",                      "INTJ"],
  "ner_iob":    ["B-PER",            "O",   "O",                      "O"],
  "meta": {"source":"wikipedia-1k","license":"CC-BY-SA"}
}
```

### Yonatan (Training & Tokenization)
**Deliverable 1**: `python -m src.training.train_model --model <name> --epsilon <value> --data <path>`  
**Deliverable 2**: `checkpoints/<model_name>_epsilon_<value>/`

**Specific Tasks:**
- **Tokenizer Implementation**: Extend GPT-2 tokenizer with special tokens
- **ε-Masking**: Apply controlled masking based on epsilon values
- **Model Training**: Train GPT-2 models with different masking levels
- **Checkpoint Management**: Save and load model states
- **Training Pipeline**: Handle batching, optimization, logging

**Tensor Dimensions**:
- Input IDs: `[batch_size, 512]`
- Attention Mask: `[batch_size, 512]`
- Labels: `[batch_size, 512]` (shifted input_ids)
- Mask Positions: `[batch_size, max_masks_per_sample]`
- Entity Positions: `[batch_size, max_entities_per_sample, 2]`

### Omer (Evaluation & Analysis)
**Deliverable 1**: `python -m src.evaluation.evaluate --model <path> --benchmark <name> --output <path>`  
**Deliverable 2**: `results/<model_name>_<benchmark>_<timestamp>/`

**Specific Tasks:**
- **Winograd Evaluation**: Test models on Winograd Schema Challenge
- **Benchmark Suite**: Implement 4-quadrant task evaluation
- **Error Analysis**: Identify wrong answers and patterns
- **Performance Comparison**: Compare across epsilon values
- **Visualization**: Create plots and reports
- **Result Reporting**: Generate comprehensive evaluation reports

## 📁 File Structure

```
src/
├── data_prep/          # Gilad's responsibility
│   ├── dataset_loader.py      # Load WikiText-103, OSCAR, custom datasets
│   ├── data_cleaner.py        # Text cleaning and normalization
│   ├── data_validator.py      # Validate spans, entities, formats
│   ├── data_formatter.py      # Format to exact JSON structure
│   └── cli.py                 # CLI: prepare_data command
├── training/           # Yonatan's responsibility
│   ├── tokenizer.py           # GPT-2 tokenizer with special tokens
│   ├── masking.py             # ε-masking implementation
│   ├── trainer.py             # Model training pipeline
│   ├── model_utils.py         # Model architecture utilities
│   ├── checkpoint_manager.py  # Save/load model checkpoints
│   └── cli.py                 # CLI: train_model command
└── evaluation/         # Omer's responsibility
    ├── winograd_evaluator.py  # Winograd Schema Challenge evaluation
    ├── benchmark_suite.py     # 4-quadrant benchmark tasks
    ├── error_analyzer.py      # Error pattern analysis
    ├── result_analyzer.py     # Performance comparison
    ├── visualizer.py          # Plots and visualizations
    └── cli.py                 # CLI: evaluate command
```

## 📚 Key Files

- **`not-to-know-poc.ipynb`**: Complete working demo - **START HERE**
- **`src/`**: Skeleton implementation for each team member
- **This README**: Team responsibilities and data specifications
