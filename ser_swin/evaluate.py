"""以最佳模型對驗證集推論，收集預測結果並輸出分類報告。"""
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .model import get_device, load_best_model


def predict(model: torch.nn.Module, val_loader) -> tuple[np.ndarray, np.ndarray]:
    """回傳 (all_labels, all_preds) 的 numpy 陣列。"""
    device = get_device()
    all_preds, all_labels = [], []

    print("Evaluating best model on validation set...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate(val_loader, class_names: list[str],
             verbose: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """載入最佳模型並評估驗證集。

    回傳 (all_labels, all_preds, confusion_matrix)。
    """
    device = get_device()
    model = load_best_model(device)
    all_labels, all_preds = predict(model, val_loader)

    if verbose:
        print("\n" + "=" * 40)
        print("Classification Report (Best Checkpoint)")
        print("=" * 40)
        print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    return all_labels, all_preds, cm
