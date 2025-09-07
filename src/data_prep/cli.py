"""
Command-line interface for data preprocessing.

This module provides CLI commands for dataset preparation and preprocessing.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .dataset_loader import DatasetLoader
from .data_cleaner import DataCleaner
from .data_validator import DataValidator
from .data_formatter import DataFormatter


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for Knowledge vs Reasoning Separation project"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Prepare data command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--dataset", required=True, 
                               choices=["wikitext", "oscar", "winograd", "squad", "custom"],
                               help="Dataset to prepare")
    prepare_parser.add_argument("--output", required=True, type=Path,
                               help="Output directory for processed data")
    prepare_parser.add_argument("--split", default="train",
                               choices=["train", "valid", "test", "all"],
                               help="Dataset split to process")
    prepare_parser.add_argument("--config", type=Path,
                               help="Configuration file for preprocessing")
    
    # Validate data command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--input", required=True, type=Path,
                                help="Input dataset file or directory")
    validate_parser.add_argument("--type", required=True,
                                choices=["winograd", "squad", "training"],
                                help="Type of dataset to validate")
    validate_parser.add_argument("--report", type=Path,
                                help="Output path for validation report")
    
    # Clean data command
    clean_parser = subparsers.add_parser("clean", help="Clean dataset")
    clean_parser.add_argument("--input", required=True, type=Path,
                             help="Input dataset file")
    clean_parser.add_argument("--output", required=True, type=Path,
                             help="Output file for cleaned data")
    clean_parser.add_argument("--min-length", type=int, default=10,
                             help="Minimum text length")
    clean_parser.add_argument("--max-length", type=int, default=1000,
                             help="Maximum text length")
    
    args = parser.parse_args()
    
    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "validate":
        validate_data(args)
    elif args.command == "clean":
        clean_data(args)
    else:
        parser.print_help()


def prepare_data(args):
    """
    Prepare dataset for training/evaluation.
    
    Args:
        args: Parsed command line arguments
    """
    
    
    try:
        print(f"Starting data preparation for WikiText-103 dataset...")
        print(f"Split: {args.split}")
        print(f"Output directory: {args.output}")
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load dataset using DatasetLoader
        print("Loading dataset...")
        loader = DatasetLoader()
        texts = loader.load_wikitext(split=args.split)
        print(f"Successfully loaded {len(texts)} text samples from {args.split}")
        
        # TODO: Continue with cleaning, formatting, validation, and saving
        print("Dataset loading completed. Next steps: cleaning, formatting, validation, and saving.")
        
    except Exception as e:
        print(f"Data preparation failed: {str(e)}")
        raise


def validate_data(args):
    """
    Validate dataset quality.
    
    Args:
        args: Parsed command line arguments
    """
    # TODO: Implement data validation
    # 1. Load dataset
    # 2. Run validation checks
    # 3. Generate validation report
    print(f"Validating {args.type} dataset from {args.input}")
    print("TODO: Implement data validation")


def clean_data(args):
    """
    Clean dataset.
    
    Args:
        args: Parsed command line arguments
    """
    # TODO: Implement data cleaning
    # 1. Load raw data
    # 2. Apply cleaning operations
    # 3. Save cleaned data
    print(f"Cleaning data from {args.input}")
    print(f"Output: {args.output}")
    print("TODO: Implement data cleaning")


if __name__ == "__main__":
    main()
