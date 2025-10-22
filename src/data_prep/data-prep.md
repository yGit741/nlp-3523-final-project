# Data Preparation Pipeline

This document describes the complete data preparation process for training and validation, including data structure, dataset selection, annotation, cloud storage, and execution.

## Data Structure

The data preparation pipeline processes raw text data and outputs structured JSON documents with comprehensive linguistic annotations. Each processed document contains:

### Core Fields
- **`id`**: Unique identifier for each sentence/document
- **`text`**: Original text content
- **`sent_spans`**: Sentence boundary annotations with start/end character positions
- **`punct_spans`**: Punctuation mark locations and values
- **`special_tags`**: Non-linguistic entities (NLEs) including:
  - Phone numbers (PHONE)
  - Addresses (ADDRESS) 
  - IP addresses (IP_ADDRESS)
  - Social Security Numbers (SSN)
  - URLs and email addresses
- **`ner_spans`**: Named entity recognition spans with entity IDs and labels
- **`pos_tokens`**: Individual tokens from the text
- **`pos_tags`**: Part-of-speech tags for each token
- **`ner_iob`**: IOB (Inside-Outside-Beginning) tags for named entity recognition

### Example Structure
```json
{
  "id": "sent_uuid-1234",
  "text": "John Smith works at Google in California.",
  "sent_spans": [{"start": 0, "end": 40}],
  "punct_spans": [{"start": 39, "end": 40, "value": "."}],
  "special_tags": [],
  "ner_spans": [
    {
      "entity_id": "PERSON-JOHN_SMITH",
      "start": 0,
      "end": 10,
      "label": "PERSON"
    },
    {
      "entity_id": "ORG-GOOGLE", 
      "start": 20,
      "end": 26,
      "label": "ORG"
    },
    {
      "entity_id": "GPE-CALIFORNIA",
      "start": 30,
      "end": 40,
      "label": "GPE"
    }
  ],
  "pos_tokens": ["John", "Smith", "works", "at", "Google", "in", "California", "."],
  "pos_tags": ["PROPN", "PROPN", "VERB", "ADP", "PROPN", "ADP", "PROPN", "PUNCT"],
  "ner_iob": ["B-PERSON", "I-PERSON", "O", "O", "B-ORG", "O", "B-GPE", "O"]
}
```
We defined this data structure prior to beginning the implementation to establish a common framework for collaboration. This ensured that all team members shared a unified reference point, allowing each student to independently work on different components of the project while maintaining consistency and interoperability.

## Choosing Dataset

The pipeline uses **OpenWebText** dataset from Hugging Face (`Skylion007/openwebtext`) for the following reasons:

### Dataset Selection Criteria
- **Scale**: Large-scale (approximately 40GB) web text corpus suitable for training language models. This dataset is a mock for the original dataset that was used to train GPT-2
- **Diversity**: Contains diverse text from various web sources
- **Quality**: Pre-filtered and cleaned web text content
- **Accessibility**: Available through Hugging Face datasets with streaming support

## Annotating the Dataset

The annotation process is orchestrated by the `SpacyJSONGenerator` class, which uses spaCy's transformer-based model for comprehensive linguistic analysis.

### Annotation Components

#### 1. Core NLP Pipeline
 - **Model**: `en_core_web_trf` - Transformer-based spaCy model
 
    We chose this model because it achieves high performance on both POS tagging (0.97) and NER (0.84) tasks, while remaining relatively lightweight at only 12 MB.
- **Components**: 
  - Part-of-Speech (POS) tagging
  - Named Entity Recognition (NER)
  - Sentence segmentation

#### 2. Custom Entity Extraction
The pipeline includes custom patterns for extracting non-linguistic entities:
- **Phone Numbers**: Various formats including international numbers
- **Addresses**: Street addresses and PO boxes
- **IP Addresses**: IPv4 and IPv6 addresses
- **Social Security Numbers**: US SSN patterns
- **URLs and Emails**: Web addresses and email addresses


### Performance Optimizations
Because the dataset is very large, processing and execution take a significant amount of time. To address this, we applied the following optimizations:

- **GPU Acceleration**: Optional GPU support for faster processing
- **Batch Processing**: Configurable batch sizes for optimal memory usage
- **Memory Management**: Automatic GPU memory cleanup between batches
- **Efficient File Storage:** Use of Parquet files instead of .json to reduce file size, improve read/write performance, and enable faster data loading during training

## Saving the Dataset to the Cloud

To enable the use of the annotated data, we uploaded it to a cloud storage bucket in batches of approximately 4 MB per file, using the Parquet format.

## Running the Job

We run this pipeline on colab pro with T4 GPU.


## What could we do differently?
During the development and execution of the pipeline, we encountered several issues. Most were identified and resolved; however, some remain unsolved due to limitations in the current setup and available resources.
- Data Structure – We should have chosen a more training-friendly data format, such as NumPy arrays or a similar structure, to simplify the training process. (# TODO: Yonatan)
- Cloud Batch Size – The uploaded files could have been larger, which would make data retrieval and processing during training more efficient. Smaller batch sizes increased the overhead of loading and processing each batch separately.
- Resource Limitations – We adjusted the batch sizes (both in model inference and file upload) to avoid out-of-memory (OOM) issues on the CPU and GPU. Using a larger GPU or a multi-CPU environment could potentially resolve these constraints




