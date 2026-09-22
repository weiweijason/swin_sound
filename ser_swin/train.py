"""訓練流程：類別加權 Loss + 分層學習率 (LLRD) + 餘弦退火。"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import config
from .model import get_device, save_checkpoint


def compute_class_weights(targets: list[int]) -> torch.Tensor:
    """類別權重 = 總樣本數 / (類別數 * 該類別樣本數)，緩解資料不平衡。"""
    class_counts = np.bincount(targets)
    total = len(targets)
    num_classes = len(class_counts)
    return torch.FloatTensor(total / (num_classes * class_counts))


def build_optimizer(model: torch.nn.Module) -> optim.AdamW:
    """全解凍 + 分層學習率：底層 LR 小、高層與分類頭 LR 大。"""
    for param in model.parameters():
        param.requires_grad = True
    return optim.AdamW([
        {'params': model.layers[0].parameters(), 'lr': config.LAYER_LR[0]},
        {'params': model.layers[1].parameters(), 'lr': config.LAYER_LR[1]},
        {'params': model.layers[2].parameters(), 'lr': config.LAYER_LR[2]},
        {'params': model.layers[3].parameters(), 'lr': config.LAYER_LR[3]},
        {'params': model.head.parameters(), 'lr': config.HEAD_LR},
    ], weight_decay=config.WEIGHT_DECAY)


def train(model: torch.nn.Module, train_loader, val_loader,
          epochs: int = config.EPOCHS) -> dict:
    """執行完整訓練迴圈，回傳 history 字典。"""
    device = get_device()
    model = model.to(device)

    # train_loader.dataset 是 Subset，需透過 indices 取回原始標籤
    train_subset = train_loader.dataset
    train_targets = [train_subset.dataset.targets[i] for i in train_subset.indices]
    class_weights = compute_class_weights(train_targets).to(device)
    print(f"💡 類別權重：{class_weights.cpu().numpy()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 20)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_train_loss = running_loss / len(train_subset)
        epoch_train_acc = correct_train / total_train
        scheduler.step()

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_acc'].append(epoch_val_acc)

        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

        # --- Save Best Checkpoint ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            path = save_checkpoint(model)
            print(f"⭐ 驗證準確率提升至 {best_val_acc:.4f}，最佳模型已存至 {path}")

    print("\n🎉 訓練流程全部完成！")
    return history
