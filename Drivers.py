from abc import ABC, abstractmethod
from pathlib import Path
import time
import json
import tempfile
import os
class BaseSaveDriver(ABC):
    """
    Abstract base class for save drivers.
    Allows different storage implementations (local, cloud, etc.).
    """
    
    def __init__(self, batch_size: int = 100):
        """
        Initialize the base save driver.
        
        Args:
            batch_size: Number of documents per batch
        """
        self.batch_size = batch_size
        self.current_batch = []
        self.batch_count = 0
        self.documents_processed = 0
    
    @abstractmethod
    def add_document(self, document):
        """Add a document to the current batch."""
        pass
    
    @abstractmethod
    def finalize(self):
        """Save any remaining documents and return statistics."""
        pass
    
    @abstractmethod
    def get_statistics(self):
        """Get current statistics."""
        pass
    
    @abstractmethod
    def _save_current_batch(self):
        """Abstract method to save the current batch to storage."""
        pass


class LocalSaveDriver(BaseSaveDriver):
    """
    Handles all file operations and memory management for saving processed batches to local disk.
    """
    
    def __init__(self, output_dir="processed_batches", batch_size=100):
        """
        Initialize the LocalSaveDriver.
        
        Args:
            output_dir: Directory to save processed batches
            batch_size: Number of documents per batch file 
        """
        super().__init__(batch_size)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"LocalSaveDriver initialized:")
    
    def add_document(self, document):
        """
        Add a document to the current batch.
        
        Args:
            document: Processed document to add
        """
        if document is not None:
            self.current_batch.append(document)
            self.documents_processed += 1
            
            # Save batch when it reaches the desired size
            if len(self.current_batch) >= self.batch_size:
                self._save_current_batch()
    
    def _save_current_batch(self):
        """
        Save the current batch to disk and clear memory.
        """
        if not self.current_batch:
            return
        
        self.batch_count += 1
        
        # Create filename with timestamp and batch number
        timestamp = int(time.time())
        filename = f"batch_{self.batch_count:06d}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Save batch data as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_batch, f, indent=2, ensure_ascii=False)
        
        print(f"Saved batch {self.batch_count} with {len(self.current_batch)} documents to {filepath}")
        
        # Clear current batch to free memory
        self.current_batch = []
    
    def finalize(self):
        """
        Save any remaining documents and return statistics.
        
        Returns:
            tuple: (batch_count, documents_processed)
        """
        # Save remaining documents if any
        if self.current_batch:
            self._save_current_batch()
        
        print(f"LocalSaveDriver completed:")
        print(f"  - Total batches: {self.batch_count}")
        print(f"  - Total documents: {self.documents_processed}")
        
        return self.batch_count, self.documents_processed
    
    def list_batches(self):
        """
        List all batch files in the output directory.
        
        Returns:
            List of batch file paths
        """
        if not self.output_dir.exists():
            return []
        
        batch_files = list(self.output_dir.glob("batch_*.json"))
        return sorted(batch_files)
    
    def load_batch(self, batch_file_path):
        """
        Load a batch from disk.
        
        Args:
            batch_file_path: Path to the batch JSON file
        
        Returns:
            List of processed documents
        """
        with open(batch_file_path, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
        return batch_data
    
    def get_statistics(self):
        """
        Get current statistics.
        
        Returns:
            dict: Statistics about processed data
        """
        batch_files = self.list_batches()
        total_file_size = sum(f.stat().st_size for f in batch_files)
        
        return {
            'batches_created': self.batch_count,
            'documents_processed': self.documents_processed,
            'files_on_disk': len(batch_files),
            'total_size_mb': total_file_size / (1024 * 1024),
            'current_batch_size': len(self.current_batch),
            'storage_type': 'local'
        }
class CloudSaveDriver(BaseSaveDriver):
    """
    Google Cloud Storage (GCS) implementation for saving processed batches.
    """
    
    def __init__(self, bucket_name=None, batch_size=100, progress_file="gcs_processing_progress.json"):
        """
        Initialize the CloudSaveDriver with GCS support.
        
        Args:
            bucket_name: GCS bucket name (optional, can be set in config)
            batch_size: Number of documents per batch file
            progress_file: File to store processing progress for resumption
        """
        super().__init__(batch_size)
        self.progress_file = progress_file
        self.progress_data = self._load_progress()
        
        # Import here to avoid dependency issues if GCS not installed
        try:
            from google.cloud import storage
            from google.cloud.exceptions import GoogleCloudError
            
            import os
            
            from config import Config
        except ImportError as e:
            raise ImportError(f"GCS dependencies not installed. Run: pip install google-cloud-storage. Error: {e}")
        
        # Store imports for use in methods
        self.storage = storage
        self.GoogleCloudError = GoogleCloudError
        self.config = Config 
        
        # Validate and get GCS configuration
        try:
            self.config.validate_gcs_config()
            self.bucket_name = bucket_name or self.config.GCS_BUCKET_NAME
            self.project_id = self.config.GCS_PROJECT_ID
            self.credentials = self.config.get_gcs_credentials()
        except Exception as e:
            raise ValueError(f"GCS configuration error: {e}")
        
        # Initialize GCS client
        try:
            
            # Credentials from file path
            self.client = self.storage.Client.from_service_account_json(
                self.credentials, 
                project=self.project_id
            )
            
            # Get bucket reference
            self.bucket = self.client.bucket(self.bucket_name)
            
            # Test bucket access
            if not self.bucket.exists():
                raise ValueError(f"Bucket '{self.bucket_name}' does not exist or is not accessible")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GCS client: {e}")
        
        print(f"☁️  CloudSaveDriver (GCS) initialized:")
        print(f"  - Bucket: {self.bucket_name}")
        print(f"  - Project: {self.project_id}")
        print(f"  - Batch size: {self.batch_size}")
        
        # Restore state from progress file
        if self.progress_data['documents_processed'] > 0:
            self.documents_processed = self.progress_data['documents_processed']
            self.batch_count = self.progress_data['batch_count']
            print(f"  - Resuming from: {self.documents_processed} docs, {self.batch_count} batches")
    
    def _load_progress(self):
        """Load existing progress if available."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    print(f"📋 Loaded existing progress: {progress['documents_processed']} docs, {progress['batch_count']} batches")
                    return progress
            except Exception as e:
                print(f"⚠️  Could not load progress file: {e}")
        
        return {
            'documents_processed': 0,
            'batch_count': 0,
            'last_processed_batch': None,
            'start_time': time.time()
        }
    
    def _save_progress(self):
        """Save current progress to file."""
        self.progress_data.update({
            'documents_processed': self.documents_processed,
            'batch_count': self.batch_count,
            'last_save_time': time.time()
        })
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save progress: {e}")
    
    def add_document(self, document):
        """
        Add a document to the current batch.
        
        Args:
            document: Processed document to add
        """
        if document is not None:
            self.current_batch.append(document)
            self.documents_processed += 1
            
            # Save batch when it reaches the desired size
            if len(self.current_batch) >= self.batch_size:
                self._save_current_batch()
    
    def _save_current_batch(self):
        """
        Save the current batch to GCS bucket.
        """
        if not self.current_batch:
            return
        
        save_start = time.time()
        self.batch_count += 1
        
        # Create filename with timestamp and batch number
        timestamp = int(time.time())
        filename = f"batch_{self.batch_count:06d}_{timestamp}.json"
        
        try:
            # Create temporary file for JSON data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(self.current_batch, temp_file, ensure_ascii=False)
                temp_file_path = temp_file.name
            
            blob = self.bucket.blob(filename)
            
            for attempt in range(self.config.GCS_RETRY_ATTEMPTS):
                try:
                    blob.upload_from_filename(temp_file_path, content_type='application/json')
                    
                    break
                except self.GoogleCloudError as e:
                    if attempt == self.config.GCS_RETRY_ATTEMPTS - 1:
                        raise
                    print(f"⚠️  Upload attempt {attempt + 1} failed, retrying... Error: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff

                # Configure upload settings for large files
                # blob.chunk_size = self.config.GCS_UPLOAD_CHUNK_SIZE # default is 100MB
                
                # Upload with retry logic            
            
            os.unlink(temp_file_path)  # Clean up
            
            save_time = time.time() - save_start
            file_size_mb = len(json.dumps(self.current_batch)) / 1024 / 1024
            
            print(f"☁️  Saved batch {self.batch_count} with {len(self.current_batch)} documents to gs://{self.bucket_name}/{filename}")
            print(f"   ⏱️  Upload time: {save_time:.3f}s, Size: {file_size_mb:.1f} MB, Rate: {len(self.current_batch)/save_time:.1f} docs/sec")
            
        except Exception as e:
            print(f"❌ Failed to save batch {self.batch_count} to GCS: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
            except:
                pass
            raise
        
        # Save progress after each batch
        self._save_progress()
        
        # Clear current batch to free memory
        self.current_batch = []
    
    def finalize(self):
        """
        Save any remaining documents and return statistics.
        """
        finalize_start = time.time()
        
        # Save remaining documents if any
        if self.current_batch:
            print(f"🔄 Finalizing: saving remaining {len(self.current_batch)} documents to GCS...")
            self._save_current_batch()
        
        finalize_time = time.time() - finalize_start
        print(f"✅ GCS finalization completed in {finalize_time:.3f}s")
        print(f"📊 CloudSaveDriver (GCS) completed:")
        print(f"  - Total batches: {self.batch_count}")
        print(f"  - Total documents: {self.documents_processed}")
        print(f"  - Bucket: gs://{self.bucket_name}")
        
        # Clean up progress file on successful completion
        # if os.path.exists(self.progress_file):
        #     os.remove(self.progress_file)
        #     print("🧹 Cleaned up progress file")
        
        return self.batch_count, self.documents_processed
    
    def get_statistics(self):
        """Get current statistics."""
        return {
            'batches_created': self.batch_count,
            'documents_processed': self.documents_processed,
            'current_batch_size': len(self.current_batch),
            'storage_type': 'gcs',
            'bucket_name': self.bucket_name,
            'project_id': self.project_id
        }
    
    def list_batches(self):
        """
        List all batch files in the GCS bucket.
        
        Returns:
            list: List of blob objects representing batch files
        """
        try:
            blobs = list(self.bucket.list_blobs(prefix="batch_"))
            return sorted(blobs, key=lambda x: x.name)
        except Exception as e:
            print(f"❌ Failed to list batches from GCS: {e}")
            return []
    
    def load_batch(self, blob):
        """
        Load a batch from GCS.
        
        Args:
            blob: GCS blob object or blob name
            
        Returns:
            list: List of processed documents
        """
        try:
            if isinstance(blob, str):
                blob = self.bucket.blob(blob)
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
                blob.download_to_filename(temp_file.name)
                
                # Load JSON data
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Clean up
                import os
                os.unlink(temp_file.name)
                
                return data
                
        except Exception as e:
            print(f"❌ Failed to load batch from GCS: {e}")
            return []
