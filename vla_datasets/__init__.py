"""
Datasets package for VLA with optional Sensor data
"""

# Unified dataset imports
from .unified_dataset import (
    # Core unified dataset
    UnifiedVLADataset,
    unified_collate_fn,
    create_unified_dataloader,

    # Backward compatibility aliases
    AsyncInsertionMeca500DatasetWithSensor,
    NewAsyncInsertionDataset,
    async_collate_fn_with_sensor,
    create_weighted_async_dataloader,
)

__all__ = [
    # Unified datasets (RECOMMENDED)
    'UnifiedVLADataset',
    'unified_collate_fn',
    'create_unified_dataloader',

    # Backward compatibility aliases
    'AsyncInsertionMeca500DatasetWithSensor',
    'NewAsyncInsertionDataset',
    'async_collate_fn_with_sensor',
    'create_weighted_async_dataloader',
]
