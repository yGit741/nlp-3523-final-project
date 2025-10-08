# Data Preprocessing Tasks - Gilad

## �� **Project Overview**
Implement the complete data preprocessing pipeline for the Knowledge vs Reasoning Separation project, focusing on WikiText-103 dataset preparation.

## 📋 **Main Deliverables**
1. **CLI Command**: `python -m src.data_prep prepare --dataset wikitext --output <path>`
2. **Output Directory**: `data/processed/wikitext/` with properly formatted JSONL files

## ✅ **Completed Tasks**
- [x] Create `__main__.py` file for module execution
- [x] Set up CLI structure with subcommands
- [x] Focus on WikiText-103 only (other datasets for later)
- [x] Implement basic dataset loading in `prepare_data()` function

## 🔄 **In Progress**
- [ ] Implement `load_wikitext()` method in `DatasetLoader`

## 📝 **Remaining Tasks**

### **1. Dataset Loading (`dataset_loader.py`)**
- [ ] Implement `load_wikitext()` method
  - [ ] Load WikiText-103 from Hugging Face datasets
  - [ ] Filter out short texts and empty lines
  - [ ] Handle different splits (train, valid, test)
  - [ ] Add proper error handling and logging

### **2. Text Cleaning (`data_cleaner.py`)**
- [ ] Implement `clean_dataset()` method
  - [ ] Remove HTML tags and normalize whitespace
  - [ ] Handle encoding issues (UTF-8 normalization)
  - [ ] Filter by text length (min: 10, max: 1000 characters)
  - [ ] Remove duplicates
  - [ ] Validate text quality

### **3. Advanced Processing (NEW - needs to be added)**
- [ ] **Sentence Segmentation**: Split text into sentences with character-level spans
- [ ] **Named Entity Recognition (NER)**: Extract entities using spaCy with unique IDs
- [ ] **Part-of-Speech (POS) Tagging**: Tokenize and tag with POS labels
- [ ] **IOB Tagging**: Convert NER to Inside-Outside-Beginning format
- [ ] **Punctuation Extraction**: Identify and extract punctuation marks with positions
- [ ] **Special Token Detection**: Find URLs, emails, numbers with positions

### **4. Data Validation (`data_validator.py`)**
- [ ] Implement `validate_dataset_format()` method
  - [ ] Ensure all spans are valid and non-overlapping
  - [ ] Validate required fields are present
  - [ ] Check data type consistency
  - [ ] Generate validation reports

### **5. Data Formatting (`data_formatter.py`)**
- [ ] Implement `format_training_data()` method
  - [ ] Create the exact JSON structure specified in README
  - [ ] Split data into train/validation/test sets
  - [ ] Generate metadata and manifests
- [ ] Implement `save_formatted_data()` method
  - [ ] Save as JSONL files
  - [ ] Create output directory structure

### **6. CLI Integration (`cli.py`)**
- [ ] Complete `prepare_data()` function
  - [ ] Integrate all components in the pipeline
  - [ ] Add proper error handling
  - [ ] Add progress indicators
- [ ] Implement `validate_data()` function
- [ ] Implement `clean_data()` function

## 📊 **Required Output Format**
Each processed sample must have this exact structure:
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

## �� **Testing Strategy**
1. **Unit Tests**: Test each component individually
2. **Integration Tests**: Test the complete pipeline
3. **Sample Data**: Start with small samples before full dataset
4. **Validation**: Ensure output matches required format exactly

## �� **Key Dependencies**
- `datasets` library for loading WikiText-103
- `spacy` for NER and POS tagging
- `nltk` for sentence segmentation
- `transformers` for tokenization utilities

## 🎯 **Next Immediate Steps**
1. **Test current dataset loading** with small sample
2. **Implement data cleaning** pipeline
3. **Add sentence segmentation** and NER processing
4. **Format data** to required JSON structure
5. **Validate output** against requirements

## 📝 **Notes**
- Focus on WikiText-103 only for now
- Other datasets (OSCAR, Winograd, SQuAD) will be added later
- Ensure all spans are character-level and non-overlapping
- Pay attention to the exact JSON structure required
- Add comprehensive logging throughout the pipeline

## �� **Related Files**
- `README.md` - Project overview and requirements
- `not-to-know-poc.ipynb` - Reference implementation
- `src/data_prep/` - All implementation files