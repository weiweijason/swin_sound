import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

# ==========================
# 1. 解決數據不平衡：計算類別加權 Loss (Weighted Cross-Entropy)
# ==========================
# 從 dataset 取得所有訓練資料的標籤分布
targets = dataset.targets
class_counts = np.bincount(targets)
total_samples = len(targets)
num_classes = len(class_counts)

# 計算權重 (公式：總樣本數 / (類別數量 * 該類別樣本數))
# 樣本越少的類別 (如 surprised)，算出來的懲罰權重就會越高
class_weights = total_samples / (num_classes * class_counts)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 將權重實體化進 Loss Function 中
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
print(f"💡 類別權重計算完成：\n{class_weights}")

# ==========================
# 2. 進階微調策略：全解凍與分層學習率 (LLRD)
# ==========================
# 這次我們把整座模型「全部解凍」
for param in model.parameters():
    param.requires_grad = True

# 針對不同層，設定不同的學習率 (越底層 LR 越小，越高層與分類頭 LR 越大)
optimizer = optim.AdamW([
    {'params': model.layers[0].parameters(), 'lr': 1e-6}, # Stage 1: 基礎頻譜紋理 (微調即可)
    {'params': model.layers[1].parameters(), 'lr': 5e-6}, # Stage 2
    {'params': model.layers[2].parameters(), 'lr': 1e-5}, # Stage 3: 中階特徵
    {'params': model.layers[3].parameters(), 'lr': 5e-5}, # Stage 4: 高階情緒語意
    {'params': model.head.parameters(), 'lr': 1e-4}       # 分類頭: 專注於你的 8 分類任務
], weight_decay=0.01)

# ==========================
# 3. 訓練環境與紀錄設定
# ==========================
save_dir = '/content/drive/MyDrive/SER_SwinV2_Checkpoints'
os.makedirs(save_dir, exist_ok=True)

epochs = 10
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
best_val_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

# ==========================
# 4. 開始訓練迴圈 (Training Loop)
# ==========================
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 20)

    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    train_bar = tqdm(train_loader, desc='Training')
    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # 這裡計算的 loss 已經自動包含了前面設定的 class_weights
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = correct_train / total_train
    scheduler.step()

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc='Validation')
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = correct_val / total_val

    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_acc'].append(epoch_val_acc)

    print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
    print(f"Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:.4f}")

    # --- Save Checkpoint ---
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        save_path = os.path.join(save_dir, 'swinv2_ravdess_best_8classes.pth')
        torch.save(model.state_dict(), save_path)
        print(f"⭐ 驗證準確率提升至 {best_val_acc:.4f}，最佳模型已備份至 Google Drive！")

print("\n🎉 訓練流程全部完成！")