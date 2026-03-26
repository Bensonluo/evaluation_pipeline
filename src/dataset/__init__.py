"""
Dataset Module for RAG Chatbot Evaluation Pipeline

This module provides data loading, sampling, and conversion utilities
for evaluation test datasets.

Design Reference: DESIGN.md Chapter 13 - Data Management
"""

from .loader import DatasetLoader, load_jsonl, load_dataset_from_db
from .sampler import StratifiedSampler, sample_dataset
from .converter import FormatConverter, convert_to_internal_format

__all__ = [
    # Loader
    "DatasetLoader",
    "load_jsonl",
    "load_dataset_from_db",
    # Sampler
    "StratifiedSampler",
    "sample_dataset",
    # Converter
    "FormatConverter",
    "convert_to_internal_format",
]

__version__ = "1.0.0"
