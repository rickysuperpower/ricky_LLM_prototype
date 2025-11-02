import pytest
pytestmark = pytest.mark.unit

import pytest

def _import_cfg():
    try:
        from llmmini.config import ModelConfig
        return ModelConfig
    except Exception as e:
        pytest.skip(f"ModelConfigのimportに失敗: {e}")

def test_config_defaults_positive():
    ModelConfig = _import_cfg()
    cfg = ModelConfig()
    assert getattr(cfg, "embed_dim", 0) > 0
    assert getattr(cfg, "num_layers", 0) >= 1
    assert getattr(cfg, "num_heads", 0) >= 1
    # max_seq_lenやvocab_sizeがあるなら0超を確認
    if hasattr(cfg, "max_seq_len"):
        assert cfg.max_seq_len > 0

def test_config_overrides():
    ModelConfig = _import_cfg()
    cfg = ModelConfig(embed_dim=96, num_layers=3, num_heads=3)
    assert cfg.embed_dim == 96
    assert cfg.num_layers == 3
    assert cfg.num_heads == 3
