import pytest
pytestmark = pytest.mark.integration

import pytest
import torch

def _import_model():
    try:
        from llmmini.model import DummyGPTModel
        return DummyGPTModel
    except Exception as e:
        pytest.skip(f"DummyGPTModelのimportに失敗: {e}")

def test_model_forward_logits_shape(device):
    DummyGPTModel = _import_model()
    vocab = 100
    model = DummyGPTModel(
        vocab_size=vocab, max_seq_len=16,
        embed_dim=64, num_heads=4, num_layers=2
    ).to(device)
    B, T = 2, 12
    x = torch.randint(0, vocab, (B, T), device=device)
    logits = model(x)
    assert logits.shape == (B, T, vocab)
    assert torch.isfinite(logits).all()

def test_model_backward_minimal(device):
    DummyGPTModel = _import_model()
    vocab = 50
    model = DummyGPTModel(
        vocab_size=vocab, max_seq_len=8,
        embed_dim=32, num_heads=2, num_layers=1
    ).to(device)
    x = torch.randint(0, vocab, (1, 8), device=device)
    logits = model(x)
    loss = logits.mean()
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

@pytest.mark.parametrize("T", [1, 8, 16])
def test_model_handles_edge_seq_len(device, T):
    DummyGPTModel = _import_model()
    vocab = 30
    model = DummyGPTModel(
        vocab_size=vocab, max_seq_len=max(16, T),
        embed_dim=32, num_heads=2, num_layers=1
    ).to(device)
    x = torch.randint(0, vocab, (2, T), device=device)
    logits = model(x)
    assert logits.shape == (2, T, vocab)
