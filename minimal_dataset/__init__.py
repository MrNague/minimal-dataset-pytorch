"""
Minimal Dependency Dataset Library for PyTorch.

Experimental data-loading library developed as part of a Bachelor Project
at DFKI.
"""

from .dataset import BaseDataset
from .parquet_dataset import ParquetDataset
from .dataloader import DataLoader
from .sampler import LockFreeSampler
from .monitored_queue import MonitoredQueue
from .metrics import MetricsTracker, WorkerMetrics

__version__ = "0.1.0"

__all__ = [
    "BaseDataset",
    "ParquetDataset",
    "DataLoader",
    "LockFreeSampler",
    "MonitoredQueue",
    "MetricsTracker",
    "WorkerMetrics",
]
