#!/usr/bin/env python3
"""
Test script to demonstrate the GCS CloudSaveDriver functionality.
"""

import os
from spacy_pipe_updated import SpacyJSONGenerator
from Drivers import CloudSaveDriver, LocalSaveDriver
from datasets import load_dataset
from config import Config

def test_gcs_driver():
    """Test the GCS CloudSaveDriver with a small dataset."""
    print("☁️  Testing GCS CloudSaveDriver")
    print("=" * 50)
    
    # Check if GCS credentials are configured
    try:
        Config.validate_gcs_config()
        print("✅ GCS configuration is valid")
    except Exception as e:
        print(f"❌ GCS configuration error: {e}")
        return
    
    # Initialize generator
    generator = SpacyJSONGenerator(batch_size=50, n_process=1,require_gpu=True)
    
    # Load a small sample of OpenWebText
    print("📥 Loading OpenWebText dataset...")
    dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, streaming=True)
    
    # Initialize GCS CloudSaveDriver
    print("\n☁️  Initializing GCS CloudSaveDriver...")
    try:
        gcs_save_driver = CloudSaveDriver(
            batch_size=50  # Small batch size for testing
        )
    except Exception as e:
        print(f"❌ Failed to initialize GCS driver: {e}")
        raise e
    
    print("\n🚀 Starting GCS processing with detailed timing...")
    print("-" * 50)
    
    # Process with GCS timing analysis
    try: 
        result_driver = generator.process_and_save(
            dataset=dataset,
            save_driver=gcs_save_driver
            # num_batches=1  # Process only 2 batches for testing
        )
        
        print("\n📊 GCS Final Results:")
        print("-" * 30)
        stats = result_driver.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # List created files in GCS
        print(f"\n☁️  Created batch files in GCS:")
        batch_files = result_driver.list_batches()
        for batch_file in batch_files:
            size_mb = batch_file.size / 1024 / 1024 if hasattr(batch_file, 'size') else 'unknown'
            print(f"  - {batch_file.name} ({size_mb:.1f} MB)")
        
        # Example: Load and inspect a GCS batch
        if batch_files:
            print(f"\n🔍 Loading first GCS batch for inspection...")
            first_batch = result_driver.load_batch(batch_files[0])
            print(f"First batch contains {len(first_batch)} processed documents")
            if first_batch:
                print(f"Sample document structure:")
                import json
                print(json.dumps(first_batch[0], indent=2)[:500] + "...")
        
        print(f"\n✅ GCS processing test completed!")
        
    except Exception as e:
        print(f"❌ GCS processing failed: {e}")
        print("💡 Make sure your GCS credentials have the necessary permissions:")
        print("   - Storage Object Creator role for the bucket")
        print("   - Storage Object Viewer role for listing/reading")

def test_local_vs_gcs_comparison():
    """Compare LocalSaveDriver vs CloudSaveDriver performance."""
    print("\n🔄 Local vs GCS Performance Comparison")
    print("=" * 50)
    
    # Initialize generator
    generator = SpacyJSONGenerator(batch_size=50, n_process=2)
    dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True, streaming=True)
    
    # Test LocalSaveDriver
    print("\n1. Testing LocalSaveDriver...")
    local_driver = LocalSaveDriver(output_dir="comparison_local", batch_size=25)
    
    import time
    start_time = time.time()
    local_result = generator.process_and_save(dataset=dataset, save_driver=local_driver, num_batches=1)
    local_time = time.time() - start_time
    
    print(f"Local processing: {local_time:.2f}s")
    
    # Test CloudSaveDriver (if configured)
    try:
        Config.validate_gcs_config()
        print("\n2. Testing CloudSaveDriver...")
        gcs_driver = CloudSaveDriver(batch_size=200)
        
        start_time = time.time()
        gcs_result = generator.process_and_save(dataset=dataset, save_driver=gcs_driver, num_batches=1)
        gcs_time = time.time() - start_time
        
        print(f"GCS processing: {gcs_time:.2f}s")
        print(f"GCS overhead: {((gcs_time - local_time) / local_time * 100):.1f}%")
        
    except Exception as e:
        print(f"⚠️  Skipping GCS comparison: {e}")

if __name__ == "__main__":
    test_gcs_driver()
    # test_local_vs_gcs_comparison()
