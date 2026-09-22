"""視覺化：學習曲線、混淆矩陣、各類別指標長條圖。

所有圖表會存成 PNG 到 outputs/ 目錄（Colab 中 plt.show() 也可用）。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report

from . import config


def _save_fig(fig, name: str) -> Path:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"📊 圖表已存至 {path}")
    plt.close(fig)
    return path


def plot_learning_curves(history: dict) -> Path:
    """繪製 Training / Validation 的 Loss 與 Accuracy 曲線。"""
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_range, history['train_loss'], label='Training Loss', marker='o')
    axes[0].plot(epochs_range, history['val_loss'], label='Validation Loss', marker='o')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history['train_acc'], label='Training Accuracy', marker='o')
    axes[1].plot(epochs_range, history['val_acc'], label='Validation Accuracy', marker='o')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    return _save_fig(fig, "learning_curves.png")


def plot_confusion_matrix(all_labels, all_preds, class_names: list[str],
                          normalized: bool = True) -> Path:
    """繪製（正規化）混淆矩陣熱力圖。"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    fmt = '.2f' if normalized else 'd'
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'{"Normalized " if normalized else ""}Confusion Matrix '
                 f'({len(class_names)} Emotions)', fontsize=14)
    ax.set_ylabel('True Emotion', fontsize=12)
    ax.set_xlabel('Predicted Emotion', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    return _save_fig(fig, "confusion_matrix.png")


def plot_metrics_by_class(all_labels, all_preds, class_names: list[str]) -> Path:
    """繪製各類別 Precision / Recall / F1-Score 長條圖。"""
    report_dict = classification_report(all_labels, all_preds,
                                        target_names=class_names, output_dict=True)
    # 排除 accuracy / macro avg / weighted avg 三行與最後的 'support' 欄
    metrics_df = pd.DataFrame(report_dict).transpose().iloc[:-3, :-1]

    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, metrics_df['precision'], width, label='Precision', color='#4C72B0')
    ax.bar(x, metrics_df['recall'], width, label='Recall', color='#DD8452')
    ax.bar(x + width, metrics_df['f1-score'], width, label='F1-Score', color='#55A868')

    ax.set_title('Performance Metrics by Emotion Class', fontsize=14)
    ax.set_xlabel('Emotion Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(x, class_names, rotation=45)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    return _save_fig(fig, "metrics_by_class.png")
