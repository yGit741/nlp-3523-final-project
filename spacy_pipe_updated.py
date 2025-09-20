import spacy
import json
import re
import time
from typing import List, Dict, Any, Iterator, Union
from collections import defaultdict
import uuid
from pathlib import Path
from Drivers import BaseSaveDriver


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
        self.nlp = spacy.load("en_core_web_trf", disable=[ "attribute_ruler", "lemmatizer", "transformer"])
        self.batch_size = batch_size
        self.n_process = n_process
        
        
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
    
    def _extract_special_tags(self, text: str) -> List[Dict[str, Any]]:
        """Extract special tags like URLs, emails, etc."""
        special_tags = []
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        for match in re.finditer(url_pattern, text):
            special_tags.append({
                "start": match.start(),
                "end": match.end(),
                "type": "URL",
                "value": match.group()
            })
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            special_tags.append({
                "start": match.start(),
                "end": match.end(),
                "type": "EMAIL",
                "value": match.group()
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
        
        # Extract special tags (URLs, emails, etc.)
        special_tags = self._extract_special_tags(original_text)
        
        # Extract named entity spans with entity IDs
        ner_spans = []
        entity_counter = 1
        for ent in doc.ents:
            ner_spans.append({
                "entity_id": f"e{entity_counter:04d}",
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
            entity_counter += 1
        
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
    
    def process_texts_batch(self, texts: List[str], 
                           base_ids: List[str] = None,
                           meta_list: List[Dict[str, str]] = None) -> List[List[Dict[str, Any]]]:
        """Process a batch of texts, splitting each into sentences."""
        if base_ids is None:
            base_ids = [f"doc_{i:07d}" for i in range(len(texts))]
        
        if meta_list is None:
            meta_list = [{"source": "custom", "license": "CC-BY-SA"}] * len(texts)
        
        # Process all texts in batch
        docs = list(self.nlp.pipe(texts, batch_size=self.batch_size, n_process=self.n_process))
        
        # Extract sentences and process
        all_results = []
        for doc, base_id, meta in zip(docs, base_ids, meta_list):
            text_results = []
            for i, sent in enumerate(doc.sents):
                sentence_id = f"{base_id}_{i:07d}"
                # Create a new doc for just this sentence for processing
                sent_doc = self.nlp(sent.text)
                result = self.process_single_doc(sent_doc, sent.text, sentence_id)
                text_results.append(result)
            all_results.append(text_results)
        
        return all_results
    
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

    def process_and_save(self, dataset, save_driver: BaseSaveDriver, num_batches=None):
        """
        Process dataset using Hugging Face map() function with configurable save driver.
        Includes detailed timing measurements and bottleneck analysis.
        
        Args:
            dataset: Hugging Face dataset
            save_driver: SaveDriver instance for handling storage (local, cloud, etc.)
            num_batches: Number of batches to process (None = process all)
        
        Returns:
            BaseSaveDriver: The save driver instance with statistics
        """
        import time
        import psutil
        import os
        
        # Initialize timing and performance tracking
        total_start_time = time.time()
        timing_stats = {
            'hf_map_time': 0,
            'spacy_processing_time': 0,
            'save_operations_time': 0,
            'iteration_time': 0,
            'total_documents': 0,
            'total_batches_processed': 0,
            'memory_usage': []
        }
        
        print(f"🚀 Starting HF map() optimized processing with {save_driver.__class__.__name__}...")
        print(f"📊 Initial memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")
        
        def process_batch_texts(batch):
            """Process a batch of texts with spaCy using HF map."""
            batch_start = time.time()
            
            # Text filtering
            filter_start = time.time()
            texts = [text for text in batch['text'] if len(text) >= 10]
            filter_time = time.time() - filter_start
            
            if not texts:
                return {'processed': [None] * len(batch['text'])}
            
            try:
                # Process batch with spaCy
                spacy_start = time.time()
                processed_docs = self.process_sentences_batch(texts)
                spacy_time = time.time() - spacy_start
                
                # Pad with None for filtered texts
                padding_start = time.time()
                result = []
                text_idx = 0
                for text in batch['text']:
                    if len(text) >= 10:
                        result.append(processed_docs[text_idx])
                        text_idx += 1
                    else:
                        result.append(None)
                padding_time = time.time() - padding_start
                
                batch_total_time = time.time() - batch_start
                
                # Log batch processing details (every 10th batch to avoid spam)
                if hasattr(process_batch_texts, 'batch_count'):
                    process_batch_texts.batch_count += 1
                else:
                    process_batch_texts.batch_count = 1
                
                if process_batch_texts.batch_count % 10 == 0:
                    print(f"  📦 Batch {process_batch_texts.batch_count}: {len(texts)} texts processed in {batch_total_time:.3f}s")
                    print(f"    - Filtering: {filter_time:.3f}s, SpaCy: {spacy_time:.3f}s, Padding: {padding_time:.3f}s")
                    print(f"    - Rate: {len(texts)/batch_total_time:.1f} texts/sec")
                
                return {'processed': result}
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
        timing_stats['hf_map_time'] = hf_map_time
        print(f"✅ HF map() setup completed in {hf_map_time:.3f}s")
        
        # Process and save data using the provided save driver
        print("💾 Processing and saving data...")
        iteration_start = time.time()
        processed_count = 0
        last_progress_time = time.time()
        
        for example in processed_dataset:
            doc_start = time.time()
            
            # Add document to save driver (handles memory management)
            save_start = time.time()
            save_driver.add_document(example['processed'])
            save_time = time.time() - save_start
            timing_stats['save_operations_time'] += save_time
            
            processed_count += 1
            timing_stats['total_documents'] = processed_count
            
            # Check batch count more frequently to respect num_batches limit
            current_batch_count = save_driver.batch_count
            timing_stats['total_batches_processed'] = current_batch_count
            
            # Progress update every 1000 documents with detailed timing
            if processed_count % 1000 == 0:
                current_time = time.time()
                elapsed_since_last = current_time - last_progress_time
                docs_per_sec = 1000 / elapsed_since_last
                
                stats = save_driver.get_statistics()
                memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                timing_stats['memory_usage'].append(memory_mb)
                
                print(f"📈 Progress: {processed_count} documents processed, {stats['batches_created']} batches saved")
                print(f"   ⏱️  Rate: {docs_per_sec:.1f} docs/sec, Memory: {memory_mb:.1f} MB")
                print(f"   💾 Save operations: {timing_stats['save_operations_time']:.3f}s total")
                
                last_progress_time = current_time
            
            # Check if we've processed enough batches (check after each document)
            if num_batches is not None and current_batch_count >= num_batches:
                print(f"🛑 Reached target of {num_batches} batches. Stopping...")
                break
            
            doc_time = time.time() - doc_start
            timing_stats['iteration_time'] += doc_time
        
        iteration_time = time.time() - iteration_start
        timing_stats['iteration_time'] = iteration_time
        
        # Finalize and get statistics
        finalize_start = time.time()
        batch_count, documents_processed = save_driver.finalize()
        finalize_time = time.time() - finalize_start
        timing_stats['save_operations_time'] += finalize_time
        
        # Calculate total time and performance metrics
        total_time = time.time() - total_start_time
        
        print(f"\n🎉 Processing completed!")
        print(f"📊 Performance Summary:")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   📄 Documents processed: {documents_processed}")
        print(f"   📦 Batches created: {batch_count}")
        print(f"   🚀 Overall rate: {documents_processed/total_time:.1f} docs/sec")
        
        print(f"\n🔍 Detailed Timing Breakdown:")
        print(f"   🔄 HF map() setup: {timing_stats['hf_map_time']:.3f}s ({timing_stats['hf_map_time']/total_time*100:.1f}%)")
        print(f"   💾 Save operations: {timing_stats['save_operations_time']:.3f}s ({timing_stats['save_operations_time']/total_time*100:.1f}%)")
        print(f"   🔁 Iteration overhead: {timing_stats['iteration_time']:.3f}s ({timing_stats['iteration_time']/total_time*100:.1f}%)")
        
        if timing_stats['memory_usage']:
            print(f"\n💾 Memory Usage:")
            print(f"   📈 Peak memory: {max(timing_stats['memory_usage']):.1f} MB")
            print(f"   📉 Average memory: {sum(timing_stats['memory_usage'])/len(timing_stats['memory_usage']):.1f} MB")
        
        # Bottleneck analysis
        print(f"\n🔍 Bottleneck Analysis:")
        if timing_stats['save_operations_time'] / total_time > 0.3:
            print(f"   ⚠️  Save operations are the bottleneck ({timing_stats['save_operations_time']/total_time*100:.1f}% of time)")
            print(f"      💡 Consider increasing LocalSaveDriver.batch_size to reduce I/O frequency")
        elif timing_stats['iteration_time'] / total_time > 0.2:
            print(f"   ⚠️  Iteration overhead is significant ({timing_stats['iteration_time']/total_time*100:.1f}% of time)")
            print(f"      💡 Consider processing larger batches or optimizing the loop")
        else:
            print(f"   ✅ No major bottlenecks detected - processing is well balanced")
        
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