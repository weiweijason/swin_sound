import torch
import os
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# ==========================================
# 1. 準備模型與載入備份權重
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的運算資源: {device}")

# 建立一個空的 SwinV2 模型 (設定 num_classes=8)
# 注意：這裡設 pretrained=False 因為我們等一下要直接覆蓋權重，節省下載時間
model = timm.create_model('swinv2_base_window8_256', pretrained=False, num_classes=8)

# 讀取 Google Drive 上的最佳權重
# 使用 map_location=device 確保就算現在沒有 GPU，也能用 CPU 跑完分析
save_dir = '/content/drive/MyDrive/SER_SwinV2_Checkpoints'
best_model_path = os.path.join(save_dir, 'swinv2_ravdess_best_8classes.pth')

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print("✅ 成功載入斷線前儲存的最佳模型權重！")
else:
    print("❌ 找不到模型權重！請檢查 Google Drive 裡是否真的有存到檔案。")

model = model.to(device)
model.eval()

# ==========================================
# 2. 對 Validation Set 進行推論以收集預測結果
# ==========================================
all_preds = []
all_labels = []

# 確保你的環境裡還有 val_loader (如果 Colab 完全重啟，你需要先重新跑前面建立 DataLoader 的儲存格)
print("正在使用最佳模型評估驗證集...")
with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc='Validation Inference'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# ==========================================
# 3. 輸出分類報告與視覺化圖表
# ==========================================
class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

print("\n" + "="*40)
print("Classification Report (Best Checkpoint)")
print("="*40)
print(classification_report(all_labels, all_preds, target_names=class_names))

# 畫出正規化混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix (SwinV2 Base - Restored Model)', fontsize=14)
plt.ylabel('True Emotion', fontsize=12)
plt.xlabel('Predicted Emotion', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 畫出指標長條圖
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :-1]

x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, metrics_df['precision'], width, label='Precision', color='#4C72B0')
plt.bar(x, metrics_df['recall'], width, label='Recall', color='#DD8452')
plt.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', color='#55A868')

plt.title('Performance Metrics by Emotion Class (Restored Model)', fontsize=14)
plt.xlabel('Emotion Classes', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(x, class_names, rotation=45)
plt.legend(loc='lower right')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()