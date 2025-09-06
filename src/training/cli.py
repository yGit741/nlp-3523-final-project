"""
Command-line interface for model training.

This module provides CLI commands for training models with different
epsilon masking levels.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .trainer import ModelTrainer
from .model_utils import ModelUtils
from .checkpoint_manager import CheckpointManager


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model training for Knowledge vs Reasoning Separation project"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train model command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", required=True, 
                             choices=["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"],
                             help="Model to train")
    train_parser.add_argument("--data", required=True, type=Path,
                             help="Path to training data")
    train_parser.add_argument("--epsilon", required=True, type=float,
                             help="Epsilon masking value (0.0-1.0)")
    train_parser.add_argument("--epochs", type=int, default=3,
                             help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8,
                             help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=5e-4,
                             help="Learning rate")
    train_parser.add_argument("--output", type=Path,
                             help="Output directory for checkpoints")
    
    # Train multiple epsilons command
    train_all_parser = subparsers.add_parser("train-all", help="Train models with multiple epsilon values")
    train_all_parser.add_argument("--model", required=True,
                                 choices=["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"],
                                 help="Model to train")
    train_all_parser.add_argument("--data", required=True, type=Path,
                                 help="Path to training data")
    train_all_parser.add_argument("--epsilons", nargs="+", type=float,
                                 default=[0.0, 0.1, 0.3, 0.5],
                                 help="List of epsilon values to train")
    train_all_parser.add_argument("--epochs", type=int, default=3,
                                 help="Number of training epochs")
    train_all_parser.add_argument("--batch-size", type=int, default=8,
                                 help="Training batch size")
    
    # List checkpoints command
    list_parser = subparsers.add_parser("list", help="List available checkpoints")
    list_parser.add_argument("--model", help="Filter by model name")
    
    # Load checkpoint command
    load_parser = subparsers.add_parser("load", help="Load model checkpoint")
    load_parser.add_argument("--checkpoint", required=True, type=Path,
                            help="Path to checkpoint")
    load_parser.add_argument("--output", type=Path,
                            help="Output path for loaded model")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args)
    elif args.command == "train-all":
        train_all_models(args)
    elif args.command == "list":
        list_checkpoints(args)
    elif args.command == "load":
        load_checkpoint(args)
    else:
        parser.print_help()


def train_model(args):
    """
    Train a single model with specific epsilon.
    
    Args:
        args: Parsed command line arguments
    """
    # TODO: Implement single model training
    # 1. Load training data
    # 2. Initialize model and trainer
    # 3. Train model with specified epsilon
    # 4. Save checkpoint
    print(f"Training {args.model} with epsilon={args.epsilon}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print("TODO: Implement model training")


def train_all_models(args):
    """
    Train models with multiple epsilon values.
    
    Args:
        args: Parsed command line arguments
    """
    # TODO: Implement multiple epsilon training
    # 1. Load training data
    # 2. For each epsilon value:
    #    - Initialize model and trainer
    #    - Train model
    #    - Save checkpoint
    print(f"Training {args.model} with epsilons: {args.epsilons}")
    print(f"Data: {args.data}")
    print("TODO: Implement multiple epsilon training")


def list_checkpoints(args):
    """
    List available checkpoints.
    
    Args:
        args: Parsed command line arguments
    """
    # TODO: Implement checkpoint listing
    # 1. Initialize checkpoint manager
    # 2. List checkpoints
    # 3. Display results
    print("Available checkpoints:")
    print("TODO: Implement checkpoint listing")


def load_checkpoint(args):
    """
    Load model checkpoint.
    
    Args:
        args: Parsed command line arguments
    """
    # TODO: Implement checkpoint loading
    # 1. Load checkpoint
    # 2. Save to output path if specified
    print(f"Loading checkpoint: {args.checkpoint}")
    print("TODO: Implement checkpoint loading")


if __name__ == "__main__":
    main()
