# llmmini/tokenizer_bpe.py

import tiktoken
from typing import List


class BPETokenizer:
    """
    tiktoken を使った GPT-2 互換 BPE トークナイザ
    - encode(text) -> List[int]
    - decode(List[int]) -> str
    - vocab_size: encoding の語彙数（50257 のはず）
    """

    def __init__(self, encoding_name: str = "gpt2"):
        """
        encoding_name:
            "gpt2"       : GPT-2 / GPT-2-small などと同じ 50,257 語彙の BPE
            "cl100k_base": ChatGPT / GPT-4 系の BPE
        """
        # tiktoken のエンコーディングを取得
        self.encoding = tiktoken.get_encoding(encoding_name)

        # GPTModel の埋め込み次元数に必要 → 語彙数
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> List[int]:
        """
        テキスト → トークンID のリスト
        """
        token_ids = self.encoding.encode(
            text,
            allowed_special={"<|endoftext|>"}  # （必要なら特殊トークン拡張できる）
        )
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        トークンIDのリスト → テキスト
        """
        text = self.encoding.decode(token_ids)
        return text
