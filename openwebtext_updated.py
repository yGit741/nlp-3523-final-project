import os
import json
import time
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from spacy_pipe_updated import SpacyJSONGenerator
from Drivers import LocalSaveDriver

# Initialize generator
generator = SpacyJSONGenerator(batch_size=100, n_process=1)
dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, streaming=True)

def save_batch_to_disk(batch_data, batch_number, output_dir="processed_batches"):
    """
    Save a batch of processed data to disk as JSON.
    
    Args:
        batch_data: List of processed documents
        batch_number: Batch number for file naming
        output_dir: Directory to save batches
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp and batch number
    timestamp = int(time.time())
    filename = f"batch_{batch_number:06d}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    # Save batch data as JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved batch {batch_number} with {len(batch_data)} documents to {filepath}")
    return filepath

def process_and_save_batches(dataset, generator, num_batches=None, batch_size=100, output_dir="processed_batches"):
    """
    Process OpenWebText dataset in batches and save to disk using streaming processing.
    
    Args:
        dataset: OpenWebText dataset
        generator: SpacyJSONGenerator instance
        num_batches: Number of batches to process (None = process all available data)
        batch_size: Number of documents per batch
        output_dir: Directory to save processed batches
    """
    if num_batches is None:
        print(f"Starting to process ALL available data in batches of {batch_size} documents each...")
    else:
        print(f"Starting to process {num_batches} batches of {batch_size} documents each...")
    
    # Get the training split (OpenWebText typically has 'train' split)
    train_dataset = dataset['train']
    
    batch_count = 0
    documents_processed = 0
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def text_generator():
        """Generator that yields texts from the dataset."""
        for item in train_dataset:
            text = item.get('text', '')
            # Skip empty or very short texts
            if len(text) >= 10:
                yield text
            else:
                print(f"Skipping empty or very short text: {text}")
    
    def sentence_id_generator():
        """Generator that yields unique sentence IDs."""
        doc_count = 0
        sent_count = 0
        while True:
            yield f"doc_{doc_count:07d}_sent_{sent_count:07d}__{str(datetime.now())}"
            sent_count += 1
            if sent_count >= batch_size:
                doc_count += 1
                sent_count = 0
    
    # Use streaming processing
    current_batch = []
    
    try:
        for processed_doc in generator.process_sentences_streaming(
            sentences=text_generator(),
            sentence_id_generator=sentence_id_generator()
        ):
            current_batch.append(processed_doc)
            
            # Save batch when it reaches the desired size
            if len(current_batch) >= batch_size:
                print(f"Processing batch {batch_count + 1}...")
                
                # Save batch to disk
                save_batch_to_disk(current_batch, batch_count + 1, output_dir)
                
                # Update counters
                batch_count += 1
                documents_processed += len(current_batch)
                
                # Clear current batch
                current_batch = []
                
                # Check if we've processed enough batches (only if num_batches is specified)
                if num_batches is not None and batch_count >= num_batches:
                    print(f"Reached target of {num_batches} batches. Stopping...")
                    break
                
                # Progress update every 10 batches
                if batch_count % 10 == 0:
                    print(f"Progress: {batch_count} batches completed, {documents_processed} documents processed")
    
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user after {batch_count} batches")
    
    # Process remaining documents if any
    if current_batch:
        print(f"Processing final batch {batch_count + 1}...")
        save_batch_to_disk(current_batch, batch_count + 1, output_dir)
        batch_count += 1
        documents_processed += len(current_batch)
    
    print(f"Completed processing {batch_count} batches with {documents_processed} total documents")
    return batch_count, documents_processed

def load_processed_batch(batch_file_path):
    """
    Load a processed batch from disk.
    
    Args:
        batch_file_path: Path to the batch JSON file
    
    Returns:
        List of processed documents
    """
    with open(batch_file_path, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    return batch_data

def list_processed_batches(output_dir="processed_batches"):
    """
    List all processed batch files in the output directory.
    
    Args:
        output_dir: Directory containing processed batches
    
    Returns:
        List of batch file paths
    """
    batch_dir = Path(output_dir)
    if not batch_dir.exists():
        return []
    
    batch_files = list(batch_dir.glob("batch_*.json"))
    return sorted(batch_files)

# Example usage
if __name__ == "__main__":
    print("🚀 OpenWebText Processing with Configurable Save Drivers")
    print("=" * 60)
    
    # Option 1: Process with LocalSaveDriver
    print("\n🔄 Processing with LocalSaveDriver...")
    start_time = time.time()
    
    local_save_driver = LocalSaveDriver(
        output_dir="processed_batches_local", 
        batch_size=50
    )
    
    local_result = generator.process_and_save(
        dataset=dataset,
        save_driver=local_save_driver,
        num_batches=2
    )
    
    local_time = time.time() - start_time
    print(f"Local processing completed in {local_time:.2f} seconds")
    
    # Option 2: Process with CloudSaveDriver (placeholder)
    print("\n🔄 Processing with CloudSaveDriver...")
    start_time = time.time()
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"LocalSaveDriver:")
    local_stats = local_result.get_statistics()
    for key, value in local_stats.items():
        print(f"  - {key}: {value}")
    
    
    
    
    # List local files (cloud files would be in cloud storage)
    if hasattr(local_result, 'list_batches'):
        local_files = local_result.list_batches()
        print(f"\n📁 Local batch files:")
        for batch_file in local_files:
            file_size = batch_file.stat().st_size / 1024  # Size in KB
            print(f"  - {batch_file.name} ({file_size:.1f} KB)")
    
    print(f"\n🎉 Processing complete!")
    print(f"Local files saved to: processed_batches_local/")