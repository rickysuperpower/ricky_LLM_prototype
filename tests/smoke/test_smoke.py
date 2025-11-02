import pytest
pytestmark = pytest.mark.smoke

import torch

def test_torch_basic_ops():
    a = torch.randn(3, 3)
    b = torch.randn(3, 3)
    c = a @ b
    assert c.shape == (3, 3)
    assert torch.isfinite(c).all()
