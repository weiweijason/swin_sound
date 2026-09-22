# 2. 載入最佳模型權重
save_dir = '/content/drive/MyDrive/SER_SwinV2_Checkpoints'
best_model_path = os.path.join(save_dir, 'swinv2_ravdess_best_8classes.pth')
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 收集真實標籤與預測標籤
all_preds = []
all_labels = []

print("Evaluating best model on validation set...")
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 獲取類別名稱 (對應 0~7 的標籤)
class_names = dataset.classes

# 3. 印出詳細分類報告 (包含 Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# 4. 繪製混淆矩陣 (Confusion Matrix)
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (8 Emotions)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()