import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



class Config:
    """Configuration class for GCS and other settings."""
    
    # GCS Configuration
    GCS_CREDENTIALS_PATH = os.getenv('GCS_CREDENTIALS_PATH', None)
    GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME',None)
    GCS_PROJECT_ID = os.getenv('GCS_PROJECT_ID', None)
    
    
    # GCS Upload settings
    GCS_UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks for large files
    GCS_RETRY_ATTEMPTS = 3
    GCS_TIMEOUT = 60  # seconds
    
    @classmethod
    def validate_gcs_config(cls):
        """Validate GCS configuration."""
        if not cls.GCS_BUCKET_NAME or cls.GCS_BUCKET_NAME == None:
            raise ValueError("GCS_BUCKET_NAME must be set in config or environment")
        
        if not cls.GCS_PROJECT_ID or cls.GCS_PROJECT_ID == None:
            raise ValueError("GCS_PROJECT_ID must be set in config or environment")
        
        # Check if credentials are available
        if not Path(cls.GCS_CREDENTIALS_PATH).exists():
            raise ValueError("GCS credentials must be provided via valid GCS_CREDENTIALS_PATH")
    
    @classmethod
    def get_gcs_credentials(cls):
        """Get GCS credentials from config."""
        return cls.GCS_CREDENTIALS_PATH 