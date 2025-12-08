# stage1/genandprint.py

import torch
from llmmini.model import generate_text_simple

def text_to_token_ids(text, tokenizer):
    """テキスト -> (1, T) 形状の token id テンソル"""
    token_ids = tokenizer.encode(text)
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    """(1, T) or (T,) の token id テンソル -> テキスト"""
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    # pos_emb の長さをコンテキストサイズとして使う
    context_size = model.pos_emb.weight.shape[0]

    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded = token_ids_to_text(token_ids, tokenizer)
    print(decoded.replace("\n", " "))

    model.train()

# def generate_and_print_sample(model, tokenizer, device, start_context):
#     model.eval()
#     context_size = model.pos_emb.weight.shape[0]
#     encoded = text_to_token_ids(start_context, tokenizer).to(device)
#     with torch.no_grad():
#         token_ids = generate_text_simple(
#             model=model, idx=encoded, max_new_tokens=50,
#             context_size=context_size
#         )

#     decoded = token_ids_to_text(token_ids, tokenizer)
#     print(decoded_text.replace("\n", " "))
#     model.train()