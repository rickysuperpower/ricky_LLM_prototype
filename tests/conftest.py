# tests/conftest.py
from pathlib import Path
import sys
import pytest
import torch

# ① プロジェクトルートを sys.path に追加（llmmini を import しやすくする）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ② 全テストで共有する device フィクスチャ（CPU/GPU）
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

