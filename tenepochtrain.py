import torch
from pathlib import Path

from llmmini.model import GPTModel
from llmmini.config import GPT_CONFIG_124M
from llmmini.tokenizers.bpetokenizer import BPETokenizer

from stage1.trainsimple import train_model_simple
from stage1.preprocess import create_dataloaders

# =======================
# デバイス設定（GPU優先）
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

# =======================
# BPETokenizer 準備（学習不要）
# =======================
tokenizer = BPETokenizer()  # tiktoken gpt2 ラッパー

# =======================
# DataLoader（軽めの設定）
# =======================
train_loader, val_loader = create_dataloaders(
    tokenizer,
    batch_size=8,     # 2 → 8 にしても 1650 なら余裕
    context_size=256,  # 32 → 64 だがまだ軽い
)

# =======================
# 小さめ GPT 設定（4GB でもサクサク）
# =======================
small_config = GPT_CONFIG_124M.copy()
small_config["n_layers"] = 12     # 12 → 4
small_config["n_heads"]  = 12    # 12 → 4
small_config["d_model"]  = 768   # 768 → 256

model = GPTModel(small_config)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1,
)

# =======================
# 学習（まずはサクッと 1 epoch）
# =======================
num_epochs = 3

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs,
    eval_freq=50,   # 50ステップに1回だけ評価
    eval_iter=1,    # 評価も1バッチだけ
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

print("Training complete.")
print("Train losses:", train_losses)
print("Val losses:", val_losses)




# import torch

# from llmmini.model import GPTModel
# from llmmini.config import GPT_CONFIG_124M
# from llmmini.tokenizers.bpetokenizer import BPETokenizer

# from stage1.trainsimple import train_model_simple
# from stage1.preprocess import create_dataloaders


# # デバイス設定
# device = torch.device("cpu")


# # 乱数シード
# torch.manual_seed(123)

# # トークナイザ作成（必要に応じて差し替え）
# tokenizer = BPETokenizer()

# # DataLoader 作成
# train_loader, val_loader = create_dataloaders(
#     tokenizer=tokenizer,
#     batch_size=8,
#     context_size=GPT_CONFIG_124M["context_length"],  # or GPT_CONFIG_124M.context_length
# )

# # モデルとオプティマイザ
# model = GPTModel(GPT_CONFIG_124M)
# model.to(device)

# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=0.0004,
#     weight_decay=0.1,
# )

# # 学習ループ
# num_epochs = 10
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context="Every effort moves you", tokenizer=tokenizer
# )