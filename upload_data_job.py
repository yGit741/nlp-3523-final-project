#!/usr/bin/env python3

from spacy_generator import SpacyJSONGenerator
from Drivers import CloudSaveDriver
from datasets import load_dataset

def do_upload():    
    generator = SpacyJSONGenerator(batch_size=50, n_process=1,require_gpu=True)
    
    print("Loading OpenWebText dataset...")
    dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, streaming=True)
    print("Done loading dataset.")
    print("\n☁️  Initializing GCS CloudSaveDriver...")
    try:
        gcs_save_driver = CloudSaveDriver(
            batch_size=50  # Small batch size for testing
        )
        print("Done init driver.")
    except Exception as e:
        print(f"❌ Failed to initialize GCS driver: {e}")
        raise e
    
    try: 
        print("\nStarting GCS processing with detailed timing...")
        result_driver = generator.process_and_save(
            dataset=dataset,
            save_driver=gcs_save_driver
            # num_batches=1  # Process the whole dataset
        )
        
        
        # List created files in GCS
        print(f"\n☁️  Created batch files in GCS:")
        batch_files = result_driver.list_batches()
        
        # Example: Load and inspect a GCS batch
        if batch_files:
            print(f"\n🔍 Loading first GCS batch for inspection...")
            first_batch = result_driver.load_batch(batch_files[0])
            print(f"First batch contains {len(first_batch)} processed documents")
            if first_batch:
                print(f"Sample document structure:")
                import json
                print(json.dumps(first_batch[0], indent=2)[:500] + "...")
        
        print("Done upload job.")
        
    except Exception as e:
        print(f"❌ GCS processing failed: {e}")
        raise e

if __name__ == "__main__":
    do_upload()
