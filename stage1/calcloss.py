# stage1/calcloss.py

import torch
import torch.nn.functional as F


def calc_loss_batch(input_batch, target_batch, model, device):
    """1バッチ分のクロスエントロピー損失を計算する。"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # model は [B, T] -> [B, T, V] のロジットを返す想定
    logits = model(input_batch)  # (B, T, V)
    B, T, V = logits.shape

    # cross_entropy は (N, C) と (N,) を受け取るので reshape
    loss = F.cross_entropy(
        logits.view(B * T, V),
        target_batch.view(B * T),
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """DataLoader 全体（または num_batches 個）の平均損失を計算する。"""
    total_loss = 0.0

    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else: 
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(
#                 input_batch, target_batch, model, device
#             )
#             total_loss += loss.item()
#         else:
#             break
#     # return total_loss / num_batches