import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import IWSLT2017Dataset, collate_fn
from model import Transformer
import matplotlib.pyplot as plt
import os
import pandas as pd
from transformers import MarianTokenizer
from matplotlib import font_manager
import math


# -------------------------------
# Mask 函数
# -------------------------------
def create_transformer_masks(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
    tgt_mask = tgt_pad_mask & causal_mask
    return src_mask, tgt_mask


# -------------------------------
# 训练函数
# -------------------------------
def train_model(model, train_loader, val_loader, pad_idx, vocab_size, device, epochs=5, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0.0
        for src_ids, _, tgt_input, tgt_labels, _ in train_loader:
            src_ids = torch.clamp(src_ids, 0, vocab_size - 1).to(device)
            tgt_input = torch.clamp(tgt_input, 0, vocab_size - 1).to(device)
            tgt_labels = torch.clamp(tgt_labels, 0, vocab_size - 1).to(device)

            src_mask, tgt_mask = create_transformer_masks(src_ids, tgt_input, pad_idx)
            logits = model(src_ids, src_mask, tgt_input, tgt_mask)
            B, T, V = logits.shape
            loss = criterion(logits.reshape(B * T, V), tgt_labels.reshape(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for src_ids, _, tgt_input, tgt_labels, _ in val_loader:
                src_ids = torch.clamp(src_ids, 0, vocab_size - 1).to(device)
                tgt_input = torch.clamp(tgt_input, 0, vocab_size - 1).to(device)
                tgt_labels = torch.clamp(tgt_labels, 0, vocab_size - 1).to(device)

                src_mask, tgt_mask = create_transformer_masks(src_ids, tgt_input, pad_idx)
                logits = model(src_ids, src_mask, tgt_input, tgt_mask)
                B, T, V = logits.shape
                loss = criterion(logits.reshape(B * T, V), tgt_labels.reshape(B * T))
                val_total += loss.item()

        avg_val_loss = val_total / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch} Train Loss {avg_train_loss:.4f} Val Loss {avg_val_loss:.4f}")

    return train_losses, val_losses


# -------------------------------
# 主函数
# -------------------------------
def main():
    os.makedirs("../results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # -------------------------------
    # 数据和 tokenizer
    # -------------------------------
    MAX_SRC, MAX_TGT, BATCH, EPOCHS, LR, D_MODEL, N_HEADS, D_FF = 64, 32, 32, 10, 1e-4, 128, 8, 512
    tokenizer_name = "Helsinki-NLP/opus-mt-en-de"
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
    pad_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size

    train_ds = IWSLT2017Dataset('train', 'en', 'de', tokenizer_name, MAX_SRC, MAX_TGT, limit=10000)
    val_ds = IWSLT2017Dataset('validation', 'en', 'de', tokenizer_name, MAX_SRC, MAX_TGT, limit=2000)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # -------------------------------
    # 消融实验
    # -------------------------------
    configs = {
        "一层编码器解码器": (1, 1),
        "二层编码器解码器": (2, 2),
        "三层编码器解码器": (3, 3)
    }
    results_ablation = {}

    for name, (enc_layers, dec_layers) in configs.items():
        print(f"\n=== Training {name} ===")
        model = Transformer(
            src_vocab=vocab_size,
            tgt_vocab=vocab_size,
            d_model=D_MODEL,
            n_heads=N_HEADS,
            d_ff=D_FF,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            pad_idx=pad_idx,
            max_len=max(MAX_SRC, MAX_TGT)
        ).to(device)

        train_losses, val_losses = train_model(
            model, train_loader, val_loader, pad_idx, vocab_size, device, epochs=EPOCHS, lr=LR
        )

        pd.DataFrame({
            "epoch": range(1, EPOCHS + 1),
            "train_loss": train_losses,
            "val_loss": val_losses
        }).to_csv(f"../results/loss_{name}.csv", index=False)

        results_ablation[name] = (train_losses, val_losses)

    # -------------------------------
    # 正常训练（固定模型配置）
    # -------------------------------
    print("\n=== Normal Training ===")
    model_normal = Transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        enc_layers=2,
        dec_layers=2,
        pad_idx=pad_idx,
        max_len=max(MAX_SRC, MAX_TGT)
    ).to(device)

    train_losses_normal, val_losses_normal = train_model(
        model_normal, train_loader, val_loader, pad_idx, vocab_size, device, epochs=EPOCHS, lr=LR
    )

    pd.DataFrame({
        "epoch": range(1, EPOCHS + 1),
        "train_loss": train_losses_normal,
        "val_loss": val_losses_normal
    }).to_csv("../results/loss_normal.csv", index=False)

    # -------------------------------
    # 绘图：消融实验和normal分开两张图
    # -------------------------------
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 图1：消融实验对比图
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))

    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    colors = ['r', 'g', 'b']

    # 训练曲线 - 消融实验
    for i, (name, (train_losses, _)) in enumerate(results_ablation.items()):
        axes1[0].plot(range(1, EPOCHS + 1), train_losses,
                      linestyle=line_styles[i % len(line_styles)],
                      marker=markers[i % len(markers)],
                      color=colors[i % len(colors)],
                      label=f'{name}')

    all_train_losses = [loss for (train_losses, _) in results_ablation.values() for loss in train_losses]
    train_min, train_max = min(all_train_losses), max(all_train_losses)
    padding = (train_max - train_min) * 0.1
    axes1[0].set_ylim(train_min - padding, train_max + padding)
    axes1[0].set_xlabel("Epoch")
    axes1[0].set_ylabel("Loss")
    axes1[0].set_title("Ablation Study - Training Loss")
    axes1[0].grid(True)
    axes1[0].legend(loc='upper right')
    axes1[0].set_xticks(range(1, EPOCHS + 1))

    # 验证曲线 - 消融实验
    for i, (name, (_, val_losses)) in enumerate(results_ablation.items()):
        axes1[1].plot(range(1, EPOCHS + 1), val_losses,
                      linestyle=line_styles[i % len(line_styles)],
                      marker=markers[i % len(markers)],
                      color=colors[i % len(colors)],
                      label=f'{name}')

    all_val_losses = [loss for (_, val_losses) in results_ablation.values() for loss in val_losses]
    val_min, val_max = min(all_val_losses), max(all_val_losses)
    padding = (val_max - val_min) * 0.1
    axes1[1].set_ylim(val_min - padding, val_max + padding)
    axes1[1].set_xlabel("Epoch")
    axes1[1].set_ylabel("Loss")
    axes1[1].set_title("Ablation Study - Validation Loss")
    axes1[1].grid(True)
    axes1[1].legend(loc='upper right')
    axes1[1].set_xticks(range(1, EPOCHS + 1))

    plt.tight_layout()
    plt.savefig("../results/ablation_loss_curves.png")
    plt.close()
    print("消融实验结果保存为ablation_loss_curves.png")

    # 图2：Normal训练单独一张图
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))

    # 训练曲线 - Normal
    axes2[0].plot(range(1, EPOCHS + 1), train_losses_normal,
                  linestyle='-', marker='o', color='blue', linewidth=2, markersize=6,
                  label='Normal Training')
    axes2[0].set_xlabel("Epoch")
    axes2[0].set_ylabel("Loss")
    axes2[0].set_title("Normal Model - Training Loss")
    axes2[0].grid(True, linestyle='--', alpha=0.7)
    axes2[0].legend(loc='upper right')
    axes2[0].set_xticks(range(1, EPOCHS + 1))

    # 验证曲线 - Normal
    axes2[1].plot(range(1, EPOCHS + 1), val_losses_normal,
                  linestyle='-', marker='s', color='red', linewidth=2, markersize=6,
                  label='Normal Validation')
    axes2[1].set_xlabel("Epoch")
    axes2[1].set_ylabel("Loss")
    axes2[1].set_title("Normal Model - Validation Loss")
    axes2[1].grid(True, linestyle='--', alpha=0.7)
    axes2[1].legend(loc='upper right')
    axes2[1].set_xticks(range(1, EPOCHS + 1))

    plt.tight_layout()
    plt.savefig("../results/normal_loss_curves.png")
    plt.close()
    print("标准实验结果已保存为normal_loss_curves.png")

    # 图3：可选 - 所有模型对比图（包含normal）
    fig3, axes3 = plt.subplots(1, 2, figsize=(18, 7))

    # 训练曲线 - 所有模型
    for i, (name, (train_losses, _)) in enumerate(results_ablation.items()):
        axes3[0].plot(range(1, EPOCHS + 1), train_losses,
                      linestyle=line_styles[i % len(line_styles)],
                      marker=markers[i % len(markers)],
                      color=colors[i % len(colors)],
                      label=f'{name}')
    axes3[0].plot(range(1, EPOCHS + 1), train_losses_normal,
                  linestyle='-', marker='x', color='black', linewidth=2,
                  label='Normal')
    axes3[0].set_xlabel("Epoch")
    axes3[0].set_ylabel("Loss")
    axes3[0].set_title("All Models - Training Loss")
    axes3[0].grid(True)
    axes3[0].legend(loc='upper right')
    axes3[0].set_xticks(range(1, EPOCHS + 1))

    # 验证曲线 - 所有模型
    for i, (name, (_, val_losses)) in enumerate(results_ablation.items()):
        axes3[1].plot(range(1, EPOCHS + 1), val_losses,
                      linestyle=line_styles[i % len(line_styles)],
                      marker=markers[i % len(markers)],
                      color=colors[i % len(colors)],
                      label=f'{name}')
    axes3[1].plot(range(1, EPOCHS + 1), val_losses_normal,
                  linestyle='-', marker='x', color='black', linewidth=2,
                  label='Normal')
    axes3[1].set_xlabel("Epoch")
    axes3[1].set_ylabel("Loss")
    axes3[1].set_title("All Models - Validation Loss")
    axes3[1].grid(True)
    axes3[1].legend(loc='upper right')
    axes3[1].set_xticks(range(1, EPOCHS + 1))

    plt.tight_layout()
    plt.savefig("../results/all_models_loss_curves.png")
    plt.close()
    print("所有实验结果：all_models_loss_curves.png")


if __name__ == '__main__':
    main()