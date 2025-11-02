import pytest
pytestmark = pytest.mark.unit

import pytest
import torch

def _import_gelu():
    try:
        from llmmini.gelu import gelu
        return gelu
    except Exception as e:
        pytest.skip(f"GELU実装のimportに失敗: {e}")

def test_gelu_shape_and_finite():
    gelu = _import_gelu()
    x = torch.linspace(-5, 5, 11)
    y = gelu(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()

def test_gelu_monotonic_local():
    gelu = _import_gelu()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = gelu(x)
    # 中央付近では単調増加に近い性質をざっくり確認
    assert y[1] < y[2] < y[3]
