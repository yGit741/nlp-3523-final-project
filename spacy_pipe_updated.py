import spacy
import re
import time
from typing import List, Dict, Any, Iterator, Union
import uuid
from pathlib import Path
from Drivers import BaseSaveDriver
from spacy.pipeline import EntityRuler


class SpacyJSONGenerator:
    def __init__(self, batch_size: int = 100, n_process: int = 1, require_gpu: bool = False):
        """
        Initialize the generator with batching capabilities.
        
        Args:
            batch_size: Number of texts (sentences) to process in each batch
            n_process: Number of processes for parallel processing (use -1 for all cores)
        """
        # Load the transformer model
        if require_gpu:
            spacy.require_gpu() 
        self.nlp = spacy.load("en_core_web_trf", disable=["lemmatizer"])
        self.batch_size = batch_size
        self.n_process = n_process
        
        # Add EntityRuler for NLE extraction (BEFORE NER for better integration)
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        self._setup_nle_patterns(ruler)
    
    def _setup_nle_patterns(self, ruler: EntityRuler):
        """Setup patterns for Nonlinguistic Entity extraction using EntityRuler."""
        patterns = [
            # Phone patterns
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"}}]},
            {"label": "PHONE", "pattern": [{"TEXT": {"REGEX": r"\+[1-9]\d{1,14}"}}]},
            
            # Address patterns
            {"label": "ADDRESS", "pattern": [{"IS_DIGIT": True}, {"IS_ALPHA": True, "OP": "+"}, {"LOWER": {"IN": ["st", "street", "ave", "avenue", "rd", "road", "blvd", "boulevard", "dr", "drive", "ln", "lane", "ct", "court", "pl", "place"]}}]},
            {"label": "ADDRESS", "pattern": [{"LOWER": "p"}, {"TEXT": "."}, {"LOWER": "o"}, {"TEXT": "."}, {"LOWER": "box"}, {"IS_DIGIT": True}]},
            
            # IP Address patterns
            {"label": "IP_ADDRESS", "pattern": [{"TEXT": {"REGEX": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"}}]},
            {"label": "IP_ADDRESS", "pattern": [{"TEXT": {"REGEX": r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"}}]},
            
            # SSN patterns
            {"label": "SSN", "pattern": [{"IS_DIGIT": True, "LENGTH": 3}, {"TEXT": "-"}, {"IS_DIGIT": True, "LENGTH": 2}, {"TEXT": "-"}, {"IS_DIGIT": True, "LENGTH": 4}]},
            {"label": "SSN", "pattern": [{"IS_DIGIT": True, "LENGTH": 3}, {"IS_SPACE": True}, {"IS_DIGIT": True, "LENGTH": 2}, {"IS_SPACE": True}, {"IS_DIGIT": True, "LENGTH": 4}]},
            
            # URL and Email patterns (using built-ins)
            {"label": "URL", "pattern": [{"LIKE_URL": True}]},
            {"label": "EMAIL", "pattern": [{"LIKE_EMAIL": True}]}
        ]
        
        ruler.add_patterns(patterns)
    
        
    def _extract_punctuation_spans(self, text: str) -> List[Dict[str, Any]]:
        """Extract punctuation spans from text."""
        punct_spans = []
        punct_pattern = r'[^\w\s]'  # Match non-word, non-space characters
        
        for match in re.finditer(punct_pattern, text):
            punct_spans.append({
                "start": match.start(),
                "end": match.end(),
                "value": match.group()
            })
        
        return punct_spans
    
    def _extract_special_tags_from_doc(self, doc) -> List[Dict[str, Any]]:
        """Extract special tags from spaCy doc (NLEs are now in doc.ents)."""
        special_tags = []
        
        # Filter NLE entities (non-standard NER labels)
        nle_labels = {"PHONE", "ADDRESS", "IP_ADDRESS", "SSN", "URL", "EMAIL"}
        
        for ent in doc.ents:
            if ent.label_ in nle_labels:
                special_tags.append({
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": ent.label_,
                    "value": ent.text
                })
        
        return special_tags
    
    def _get_sentence_spans(self, doc, text: str) -> List[Dict[str, int]]:
        """Extract sentence spans."""
        sent_spans = []
        for sent in doc.sents:
            sent_spans.append({
                "start": sent.start_char,
                "end": sent.end_char
            })
        return sent_spans
    
    def process_single_doc(self, doc, original_text: str, sentence_id: str) -> Dict[str, Any]:
        """Process a single spaCy doc and return the JSON structure."""
        
        # Extract sentence spans
        sent_spans = self._get_sentence_spans(doc, original_text)
        
        # Extract punctuation spans
        punct_spans = self._extract_punctuation_spans(original_text)
        
        # Extract special tags (NLEs) from doc.ents
        special_tags = self._extract_special_tags_from_doc(doc)
        
        # Extract named entity spans with entity IDs (standard NER only)
        ner_spans = []
        nle_labels = {"PHONE", "ADDRESS", "IP_ADDRESS", "SSN", "URL", "EMAIL"}
        
        for ent in doc.ents:
            # Only include standard NER entities, not NLEs
            if ent.label_ not in nle_labels:
                ner_spans.append({
                    "entity_id": f"{ent.label_}-{str(ent).upper().replace(' ', '_').replace('-', '_')}",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "label": ent.label_
                })
        
        # Extract POS tokens and tags
        pos_tokens = []
        pos_tags = []
        ner_iob = []
        
        for token in doc:
            # Skip whitespace-only tokens
            if not token.text.strip():
                continue
                
            pos_tokens.append(token.text)
            pos_tags.append(token.pos_)
            
            # Determine IOB tag
            if token.ent_iob_ == 'B':
                ner_iob.append(f"B-{token.ent_type_}")
            elif token.ent_iob_ == 'I':
                ner_iob.append(f"I-{token.ent_type_}")
            else:
                ner_iob.append("O")
        
        # Build the final JSON structure
        result = {
            "id": sentence_id,
            "text": original_text,
            "sent_spans": sent_spans,
            "punct_spans": punct_spans,
            "special_tags": special_tags,
            "ner_spans": ner_spans,
            "pos_tokens": pos_tokens,
            "pos_tags": pos_tags,
            "ner_iob": ner_iob
        }
        
        return result
    
    def process_sentences_batch(self, sentences: List[str], sentence_ids: List[str] = None):
        """Process a batch of sentences efficiently."""
        if sentence_ids is None:
            sentence_ids = [f"sent_{str(uuid.uuid4())}" for _ in range(len(sentences))]
        
        # Process batch with spaCy
        docs = list(self.nlp.pipe(sentences, batch_size=self.batch_size, n_process=self.n_process))
        
        # Process each doc
        results = []
        for doc, original_text, sent_id in zip(docs, sentences, sentence_ids):
            result = self.process_single_doc(doc, original_text, sent_id)
            results.append(result)
        
        return results
    
   
    def process_sentences_streaming(self, sentences: Iterator[str],
                                   sentence_id_generator: Iterator[str] = None) -> Iterator[Dict[str, Any]]:
        """Process sentences in streaming fashion with batching."""
        sentence_batch = []
        id_batch = []
        
        for i, sentence in enumerate(sentences):
            sentence_batch.append(sentence)
            
            if sentence_id_generator:
                id_batch.append(next(sentence_id_generator))
            else:
                id_batch.append(f"sent_{i:07d}")
            
            # Process batch when it reaches batch_size
            if len(sentence_batch) >= self.batch_size:
                results = self.process_sentences_batch(sentence_batch, id_batch)
                for result in results:
                    yield result
                
                # Clear batches
                sentence_batch = []
                id_batch = []
        
        # Process remaining sentences
        if sentence_batch:
            results = self.process_sentences_batch(sentence_batch)
            for result in results:
                yield result

    def process_and_save(self, dataset, save_driver: BaseSaveDriver, num_batches=None, resume_from_progress=True):
        """
        Process dataset using Hugging Face map() function with configurable save driver.
        Includes detailed timing measurements and bottleneck analysis.
        
        Args:
            dataset: Hugging Face dataset
            save_driver: SaveDriver instance for handling storage (local, cloud, etc.)
            num_batches: Number of batches to process (None = process all)
            resume_from_progress: Whether to resume from existing progress (if available)
        
        Returns:
            BaseSaveDriver: The save driver instance with statistics
        """
        
        print(f"🚀 Starting HF map() optimized processing with {save_driver.__class__.__name__}...")
        
        # Check if we should resume from existing progress
        documents_to_skip = 0
        initial_batch_count = 0
        if resume_from_progress and hasattr(save_driver, 'progress_data'):
            progress = save_driver.progress_data
            if progress['documents_processed'] > 0:
                documents_to_skip = progress['documents_processed']
                initial_batch_count = progress['batch_count']
                print(f"🔄 Resuming from previous progress:")
                print(f"   📄 Documents already processed: {progress['documents_processed']}")
                print(f"   📦 Batches already created: {progress['batch_count']}")
                print(f"⏭️  Skipping first {documents_to_skip} documents...")
                
        def process_batch_texts(batch):
            """Process a batch of texts with spaCy using HF map."""
            
            texts = [text for text in batch['text'] if len(text) >= 10]
            
            if not texts:
                return {'processed': [None] * len(batch['text'])}
            
            try:
                processed_docs = self.process_sentences_batch(texts)
                return {'processed': processed_docs}
            except Exception as e:
                print(f"❌ Error processing batch: {e}")
                return {'processed': [None] * len(batch['text'])}
        
        # Use HF map() with batch processing
        print("🔄 Applying HF map() with batch processing...")
        hf_map_start = time.time()
        
        processed_dataset = dataset['train'].map(
            process_batch_texts,
            batched=True,
            batch_size=self.batch_size,  # Use spaCy batch size for HF map
            remove_columns=['text']  # Remove original text column
        )
        
        hf_map_time = time.time() - hf_map_start
        print(f"✅ HF map() setup completed in {hf_map_time:.3f}s")
        
        # Process and save data using the provided save driver
        print("💾 Processing and saving data...")
        
        processed_count = 0
        skipped_count = 0
        
        try:
            for example in processed_dataset:
                # Skip already processed documents
                if skipped_count < documents_to_skip:
                    skipped_count += 1
                    continue
                
                save_driver.add_document(example['processed'])            
                processed_count += 1
                
                # Check batch count more frequently to respect num_batches limit
                current_batch_count = save_driver.batch_count
                new_batches_created = current_batch_count - initial_batch_count
                
                # Check if we've processed enough NEW batches (check after each document)
                if num_batches is not None and new_batches_created >= num_batches:
                    print(f"🛑 Reached target of {num_batches} new batches. Stopping...")
                    print(f"   📊 Total batches: {current_batch_count}, New batches this run: {new_batches_created}")
                    break
                
                
                
        except KeyboardInterrupt:
            print("\n⚠️  Processing interrupted by user. Progress saved.")
            if hasattr(save_driver, '_save_progress'):
                save_driver._save_progress()
            raise
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            print("💾 Progress saved. You can resume later.")
            if hasattr(save_driver, '_save_progress'):
                save_driver._save_progress()
            raise
            
        
        # Finalize and get statistics
        batch_count, documents_processed = save_driver.finalize()
        
        # Calculate total time and performance metrics
        
        print(f"\n🎉 Processing completed!")
        print(f"📊 Performance Summary:")
        print(f"   📄 Documents processed: {documents_processed}")
        print(f"   📦 Batches created: {batch_count}")
        
        
        
        
        # Bottleneck analysis
        
        
        
        return save_driver

# Example usage and performance testing
if __name__ == "__main__":
    print("--- Remove comment to run the example --- ")
    # import time
    
    # # Initialize with batching
    # generator = SpacyJSONGenerator(batch_size=32, n_process=1)
    
    # # Example: Single sentence
    # print("=== Single Sentence Example ===")
    # example_text = "Turing—brilliant! See https://turing.org.uk... Really."
    # start_time = time.time()
    
    # result = generator.process_sentences_batch(
    #     [example_text]
    # )[0]
    
    # print(f"Time: {time.time() - start_time:.3f}s")
    # print(json.dumps(result, indent=2))
    
    # # Example: Batch processing
    # print("\n=== Batch Processing Example ===")
    # sentences = [
    #     "Hello world! This is a test.",
    #     "Natural language processing is amazing.",
    #     "SpaCy makes NLP easy and efficient.",
    #     "Visit https://spacy.io for more information.",
    #     "John Smith works at Google Inc."
    # ] * 10  # 50 sentences total
    
    # start_time = time.time()
    # results = generator.process_sentences_batch(sentences)
    # batch_time = time.time() - start_time
    
    # print(f"Batch processing {len(sentences)} sentences took: {batch_time:.3f}s")
    # print(f"Average per sentence: {batch_time/len(sentences):.4f}s")
    # print(f"First result sample:")
    # print(json.dumps(results[0], indent=2))
    
    # # Example: Streaming processing
    # print("\n=== Streaming Processing Example ===")
    # def sentence_generator():
    #     sentences = [
    #         "Streaming sentence 1.",
    #         "Streaming sentence 2.",
    #         "Streaming sentence 3 with https://example.com.",
    #     ]
    #     for sentence in sentences:
    #         yield sentence
    
    # start_time = time.time()
    # streaming_results = list(generator.process_sentences_streaming(sentence_generator()))
    # streaming_time = time.time() - start_time
    
    # print(f"Streaming processing took: {streaming_time:.3f}s")
    # print(f"Processed {len(streaming_results)} sentences")
    
    # # Progress callback example
    # print("\n=== Large Dataset with Progress ===")
    # def progress_callback(current, total):
    #     percentage = (current / total) * 100
    #     print(f"Progress: {current}/{total} ({percentage:.1f}%)")
    
    # large_sentences = ["This is sentence number {}.".format(i) for i in range(100)]
    # start_time = time.time()
    # large_results = generator.process_large_dataset(
    #     large_sentences, 
    #     progress_callback=progress_callback
    # )
    # large_time = time.time() - start_time
    
    # print(f"Large dataset processing took: {large_time:.3f}s")
    # print(f"Processed {len(large_results)} sentences")