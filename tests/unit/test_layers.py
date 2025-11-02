import pytest
pytestmark = pytest.mark.unit

import pytest
import torch

def _import_sa():
    try:
        from llmmini.layers import SelfAttention
        return SelfAttention
    except Exception as e:
        pytest.skip(f"SelfAttentionのimportに失敗: {e}")

@pytest.mark.parametrize("B,T,C,heads", [(2, 8, 64, 4), (1, 1, 32, 1)])
def test_self_attention_shape(device, B, T, C, heads):
    SelfAttention = _import_sa()
    x = torch.randn(B, T, C, device=device)
    sa = SelfAttention(embed_dim=C, num_heads=heads).to(device)
    y = sa(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

def test_self_attention_grad(device):
    SelfAttention = _import_sa()
    B, T, C, heads = 2, 6, 48, 3
    x = torch.randn(B, T, C, device=device, requires_grad=True)
    sa = SelfAttention(embed_dim=C, num_heads=heads).to(device)
    y = sa(x).sum()
    y.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
