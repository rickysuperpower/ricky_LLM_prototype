# run_generate.py
import torch

from llmmini.config import GPT_CONFIG_124M
from llmmini.bpetokenizer import BPETokenizer
from llmmini.model import GPTModel, generate_text_simple


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # トークナイザ
    tokenizer = BPETokenizer("gpt2")
    print(f"[INFO] vocab_size (tokenizer): {tokenizer.vocab_size}")

    # コンフィグ
    cfg = GPT_CONFIG_124M.copy()
    cfg["vocab_size"] = tokenizer.vocab_size
    context_size = cfg["context_length"]
    print(f"[INFO] context_length: {context_size}")

    # モデル
    model = GPTModel(cfg).to(device)
    model.eval()
    print("[INFO] GPTModel initialized")

    # ★ ここで学習済みパラメータを読むなら load_state_dict を入れる

    # プロンプト
    user_prompt = input("Enter your prompt (empty for default): ").strip()
    if user_prompt == "":
        user_prompt = "Hello, my name is"

    print(f"[INFO] prompt: {user_prompt!r}")

    start_ids = tokenizer.encode(user_prompt)
    idx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    # 生成
    max_new_tokens = 50
    print(f"[INFO] Generating {max_new_tokens} new tokens...")
    with torch.no_grad():
        out_idx = generate_text_simple(
            model,
            idx,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )

    out_ids = out_idx[0].tolist()
    generated_text = tokenizer.decode(out_ids)

    print("\n===== Generated Text =====")
    print(generated_text)
    print("==========================")


if __name__ == "__main__":
    main()
