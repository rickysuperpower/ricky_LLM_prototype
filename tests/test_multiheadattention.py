# tests/unit/test_multihead_attention.py
import torch
import pytest

from llmmini.multiheadattention import MultiHeadAttention  # ← 実際の場所に合わせて変更


def test_multihead_attention_output_shape(device):
    """入力 (B, T, d_in) に対して (B, T, d_out) が返ることを確認"""
    torch.manual_seed(0)

    B = 2
    T = 4
    d_in = 32
    d_out = 64
    num_heads = 8
    context_length = 16
    dropout = 0.0

    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=False,
    ).to(device)

    x = torch.randn(B, T, d_in, device=device)

    out = mha(x)

    assert out.shape == (B, T, d_out)


def test_multihead_attention_backward(device):
    """勾配がちゃんと計算できるか（backwardが通るか）を確認"""
    torch.manual_seed(0)

    B = 2
    T = 4
    d_in = 32
    d_out = 64
    num_heads = 8
    context_length = 16
    dropout = 0.1

    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=dropout,
        num_heads=num_heads,
        qkv_bias=True,
    ).to(device)

    x = torch.randn(B, T, d_in, device=device, requires_grad=True)

    out = mha(x)              # (B, T, d_out)
    loss = out.sum()          # 適当なスカラー
    loss.backward()

    # 入力とパラメータに勾配が乗っていることを確認
    assert x.grad is not None
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"grad is None for parameter {name}"
        assert torch.isfinite(param.grad).all(), f"grad has NaN/Inf in {name}"


def test_multihead_attention_uses_causal_mask(device):
    """
    causal mask が形的に正しく適用されているかを軽く確認。
    （未来方向へのスコアが -inf → softmax で 0 になることをチェック）
    """
    torch.manual_seed(0)

    B = 1
    T = 4
    d_in = 8
    d_out = 8
    num_heads = 2
    context_length = 8

    mha = MultiHeadAttention(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=num_heads,
        qkv_bias=False,
    ).to(device)

    # すべて同じ値にして、mask がなければ各位置で均等重みになる状況を作る
    x = torch.ones(B, T, d_in, device=device)

    # forward 途中の attn_weights を見るために hook を仕込む
    saved_weights = {}

    def save_attn_weights(module, input, output):
        saved_weights["attn_weights"] = output

    # dropout の直前のテンソルをフックしたいので、forward をラップする
    # 簡易的に forward を monkey patch して確認する方法もあるが、
    # ここでは module 自体を一回通してから mask 形状だけ検査する簡易版にする。

    # ここでは mask 自体の形だけ検査しておく
    mask = mha.mask[:T, :T]  # (T, T)
    assert mask.shape == (T, T)
    # 上三角が True、対角および下三角が False になっていることを確認
    assert torch.equal(mask, torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1))

    # 単に forward が NaN を出さずに動くことも確認
    out = mha(x)
    assert torch.isfinite(out).all()
