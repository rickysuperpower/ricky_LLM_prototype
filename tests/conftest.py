import pytest
import torch

@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture(scope="session")
def rng():
    torch.manual_seed(42)
    return torch.Generator().manual_seed(42)
