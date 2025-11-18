"""
Models package for Vision-Language-Action (VLA) with Sensor Integration.

This package provides the core model (`QwenVLAUnified`) and its essential
building blocks for multimodal learning in robotics.

Key Modules and Components:
- `unified_model`: Contains the main `QwenVLAUnified` model, which orchestrates
  the various encoders and action experts.
- `Encoder_model`: Defines various encoders for processing robot state and
  sensor data, including `RobotStateEncoder`, `SensorEncoder`, and
  `ForceAwareSensorEncoder`.
- `action_decoder`: Implements different action prediction experts, such as
  `FlowMatchingActionExpert` and `RegressionActionExpert`.
- `vl_encoder`: Encapsulates the Vision-Language (VL) feature encoding logic
  using the Qwen-VL model.
- `vl_cache`: Provides a caching mechanism for VLM features to optimize performance.
"""

# Core unified model
from .unified_model import QwenVLAUnified

# Encoder components
from .Encoder_model import (
    RobotStateEncoder,
    UnifiedGatedSensorEncoder,
)

# Backward compatibility aliases
SensorEncoder = UnifiedGatedSensorEncoder
ForceAwareSensorEncoder = UnifiedGatedSensorEncoder

# Action decoder components
from .action_decoder import (
    FlowMatchingActionExpert,
    RegressionActionExpert,
    DiffusionActionExpert,
)

# Vision-Language Encoder
from .vl_encoder import VisionLanguageEncoder


__all__ = [
    # Unified model (RECOMMENDED)
    'QwenVLAUnified',

    # Encoder components
    'RobotStateEncoder',
    'UnifiedGatedSensorEncoder',
    'SensorEncoder',  # Alias for backward compatibility
    'ForceAwareSensorEncoder',  # Alias for backward compatibility
    'VisionLanguageEncoder',

    # Action decoder components
    'FlowMatchingActionExpert',
    'RegressionActionExpert',
    'DiffusionActionExpert',
]