"""
Data formatting utilities for training and evaluation.

This module formats cleaned data into the appropriate structure
for training and evaluation pipelines.
"""

from typing import List, Dict, Optional, Union
from pathlib import Path
import json


class DataFormatter:
    """
    Formats data for training and evaluation.
    
    Handles:
    - Training data formatting
    - Evaluation data formatting
    - File output formatting
    - Metadata generation
    """
    
    def __init__(self, output_dir: Union[str, Path] = "data/processed"):
        """
        Initialize data formatter.
        
        Args:
            output_dir: Directory for formatted data output
        """
        self.output_dir = Path(output_dir)
        # TODO: Implement initialization
    
    def format_training_data(self, texts: List[str], dataset_name: str) -> Dict:
        """
        Format texts for training.
        
        Args:
            texts: List of cleaned texts
            dataset_name: Name of the dataset
            
        Returns:
            Formatted training data dictionary with structure:
            {
                "train": [formatted_samples],
                "validation": [formatted_samples], 
                "test": [formatted_samples],
                "metadata": {...}
            }
            
        Each formatted sample has structure:
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
        """
        # TODO: Implement training data formatting
        # - Create train/validation splits
        # - Add metadata
        # - Format for tokenization
        pass
    
    def format_winograd_data(self, schemas: List[Dict]) -> Dict:
        """
        Format Winograd schemas for evaluation.
        
        Args:
            schemas: List of Winograd schemas
            
        Returns:
            Formatted evaluation data
        """
        # TODO: Implement Winograd data formatting
        # - Group by difficulty
        # - Add metadata
        # - Format for evaluation pipeline
        pass
    
    def format_squad_data(self, samples: List[Dict]) -> Dict:
        """
        Format SQuAD data for evaluation.
        
        Args:
            samples: List of SQuAD samples
            
        Returns:
            Formatted evaluation data
        """
        # TODO: Implement SQuAD data formatting
        # - Group by question type
        # - Add metadata
        # - Format for evaluation pipeline
        pass
    
    def save_formatted_data(self, data: Dict, filename: str, format: str = "json") -> Path:
        """
        Save formatted data to file.
        
        Args:
            data: Formatted data dictionary
            filename: Output filename
            format: Output format ('json', 'jsonl', 'csv')
            
        Returns:
            Path to saved file
        """
        # TODO: Implement data saving
        # - Create output directory if needed
        # - Save in specified format
        # - Add metadata file
        pass
    
    def create_data_manifest(self, dataset_info: Dict) -> Dict:
        """
        Create dataset manifest with metadata.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            Manifest dictionary
        """
        # TODO: Implement manifest creation
        # - Dataset name and version
        # - Creation timestamp
        # - Statistics
        # - File locations
        pass
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            data: Dataset to split
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Dictionary with train/val/test splits
        """
        # TODO: Implement dataset splitting
        # - Ensure balanced splits
        # - Maintain data integrity
        # - Add split metadata
        pass
