# tests/test_tokenizer.py
import pytest
from llmmini.tokenizer import SimpleTokenizerV1  # ← これだけ

def test_encode_basic():
    vocab = {"hello": 0, ",": 1, "world": 2}
    tokenizer = SimpleTokenizerV1(vocab)

    ids = tokenizer.encode("hello, world")
    assert ids == [0, 1, 2]


def test_decode_basic():
    vocab = {"hello": 0, ",": 1, "world": 2}
    tokenizer = SimpleTokenizerV1(vocab)

    text = tokenizer.decode([0, 1, 2])
    assert text == "hello, world"


def test_round_trip():
    vocab = {"hello": 0, ",": 1, "world": 2}
    tokenizer = SimpleTokenizerV1(vocab)

    original = "hello, world"
    ids = tokenizer.encode(original)
    decoded = tokenizer.decode(ids)

    assert decoded == original


def test_unknown_token_raises_keyerror():
    vocab = {"hello": 0}
    tokenizer = SimpleTokenizerV1(vocab)

    with pytest.raises(KeyError):
        tokenizer.encode("unknown")




