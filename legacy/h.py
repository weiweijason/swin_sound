import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# 確保類別名稱與順序正確 (對應你的模型輸出)
# 如果你的模型輸出標籤順序不同，請依據 dataset.classes 的順序調整
class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# ==========================================
# 1. 繪製正規化混淆矩陣 (Normalized Confusion Matrix)
# ==========================================
# 計算混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
# 將數量轉換為百分比 (Row normalize)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix (SwinV2 Base - 8 Emotions)', fontsize=14)
plt.ylabel('True Emotion', fontsize=12)
plt.xlabel('Predicted Emotion', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================================
# 2. 繪製各類別評估指標長條圖 (Precision, Recall, F1-Score)
# ==========================================
# 取得以字典格式回傳的分類報告
report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

# 整理成 DataFrame 方便畫圖 (排除 accuracy, macro avg, weighted avg)
metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :-1]

# 設定長條圖寬度與位置
x = np.arange(len(class_names))
width = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x - width, metrics_df['precision'], width, label='Precision', color='#4C72B0')
plt.bar(x, metrics_df['recall'], width, label='Recall', color='#DD8452')
plt.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', color='#55A868')

plt.title('Performance Metrics by Emotion Class', fontsize=14)
plt.xlabel('Emotion Classes', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(x, class_names, rotation=45)
plt.legend(loc='lower right')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
