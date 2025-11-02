# Core configs & models
from .config import ModelConfig  # または GPTConfig（どちらかに統一）
from .model import DummyGPTModel

# Building blocks
from .activations import gelu
from .attention import SelfAttention
from .blocks import TransformerBlock
from .feedforward import FeedForward

# 公開シンボル
__all__ = [
    "ModelConfig",
    "DummyGPTModel",
    "gelu",
    "SelfAttention",
    "TransformerBlock",
    "FeedForward",
]

