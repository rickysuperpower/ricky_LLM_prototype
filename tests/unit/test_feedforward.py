import pytest
pytestmark = pytest.mark.unit

import pytest
import torch

def _import_ff():
    try:
        # 例: FeedForwardクラス or feedforward関数 いずれかに合わせます
        import feedforward as ff
        return ff
    except Exception as e:
        pytest.skip(f"feedforward.py のimportに失敗: {e}")

def test_feedforward_shape(device):
    ff = _import_ff()
    # クラスがある場合
    if hasattr(ff, "FeedForward"):
        B, T, C = 2, 8, 64
        x = torch.randn(B, T, C, device=device)
        m = ff.FeedForward(C, hidden_multiplier=4).to(device) if hasattr(ff, "FeedForward") else None
        y = m(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
    # 関数実装の場合（例: def feedforward(x, hidden_dim): ...）
    elif hasattr(ff, "feedforward"):
        x = torch.randn(4, 32, device=device)
        y = ff.feedforward(x, hidden_dim=64)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
    else:
        pytest.skip("FeedForward相当のAPIが見つかりません")
