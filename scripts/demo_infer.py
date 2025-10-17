import torch
from llmmini import GPTConfig, DummyGPTModel

def main():
    torch.manual_seed(123)
    cfg = GPTConfig(vocab_size=50257, context_length=256, emb_dim=768, n_layers=12, drop_rate=0.1)
    model = DummyGPTModel(cfg)

    batch = torch.randint(0, cfg.vocab_size, (4, cfg.context_length))
    logits = model(batch)
    print("Output shape:", logits.shape)  # (4, 256, 50257)

if __name__ == "__main__":
    main()
