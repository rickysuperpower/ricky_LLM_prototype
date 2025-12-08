# stage1/preprocess.py

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# プロジェクトルート/data/raw/the-verdict.txt を指すパス
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEXT_PATH = PROJECT_ROOT / "data" / "raw" / "the-verdict.txt"


def load_text(path: Path | str = DEFAULT_TEXT_PATH) -> str:
    """テキストファイルを読み込んで1つの文字列として返す。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return f.read()


class GPTDataset(Dataset):
    """LLM事前学習用のシンプルなデータセット（次トークン予測）。"""

    def __init__(self, token_ids: torch.Tensor, context_size: int) -> None:
        """
        token_ids: 1次元テンソル [N] （全文のトークンID列）
        context_size: 1サンプルあたりのコンテキスト長
        """
        self.token_ids = token_ids
        self.context_size = context_size

    def __len__(self) -> int:
        # 入力とラベルのペアを作れる最大インデックス
        return len(self.token_ids) - self.context_size

    def __getitem__(self, idx: int):
        x = self.token_ids[idx : idx + self.context_size]
        y = self.token_ids[idx + 1 : idx + 1 + self.context_size]
        return x, y


def create_dataloaders(
    tokenizer,
    batch_size: int = 8,
    context_size: int = 128,
    val_fraction: float = 0.1,
    text_path: Path | str = DEFAULT_TEXT_PATH,
) -> Tuple[DataLoader, DataLoader]:
    """
    the-verdict.txt を読み込んで、train_loader と val_loader を作成する。

    tokenizer: 既に作成済みのトークナイザ（.encode(text) を持つ）
    """
    text = load_text(text_path)

    # トークンID列に変換（あなたのトークナイザ仕様に合わせてここだけ調整）
    token_ids_list = tokenizer.encode(text)
    token_ids = torch.tensor(token_ids_list, dtype=torch.long)

    # train / val に分割
    split_idx = int(len(token_ids) * (1 - val_fraction))
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]

    train_ds = GPTDataset(train_ids, context_size)
    val_ds = GPTDataset(val_ids, context_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
