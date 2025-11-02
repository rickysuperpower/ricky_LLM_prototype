import pytest
pytestmark = pytest.mark.integration

import pytest
import torch

def _import_block():
    try:
        from llmmini.blocks import TransformerBlock
        return TransformerBlock
    except Exception as e:
        pytest.skip(f"TransformerBlockのimportに失敗: {e}")

def test_block_forward_shape(device):
    TransformerBlock = _import_block()
    B, T, C, heads = 2, 8, 64, 4
    x = torch.randn(B, T, C, device=device)
    block = TransformerBlock(embed_dim=C, num_heads=heads, mlp_ratio=4).to(device)
    y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

def test_block_backward(device):
    TransformerBlock = _import_block()
    B, T, C, heads = 1, 4, 32, 2
    x = torch.randn(B, T, C, device=device, requires_grad=True)
    block = TransformerBlock(embed_dim=C, num_heads=heads, mlp_ratio=2).to(device)
    y = block(x).mean()
    y.backward()
    assert x.grad is not None
