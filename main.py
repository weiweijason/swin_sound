"""專案入口：以子命令執行各階段。

用法：
    python main.py download      # 下載 RAVDESS / TESS / CREMA-D
    python main.py preprocess    # wav -> mel 語譜圖
    python main.py train         # 訓練 SwinV2
    python main.py evaluate      # 載入最佳模型並評估 + 畫圖
    python main.py all           # 依序執行以上全部流程
"""
import argparse

from ser_swin import config
from ser_swin.data import build_dataloaders, build_melspec_dataset, download_all
from ser_swin.evaluate import evaluate
from ser_swin.model import build_model, get_device
from ser_swin.train import train
from ser_swin.visualize import (plot_confusion_matrix, plot_learning_curves,
                                plot_metrics_by_class)


def cmd_download() -> None:
    download_all()


def cmd_preprocess() -> None:
    build_melspec_dataset()


def cmd_train() -> None:
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, dataset = build_dataloaders()
    model = build_model(pretrained=True, num_classes=len(dataset.classes))

    history = train(model, train_loader, val_loader)

    # 訓練完立即畫學習曲線
    plot_learning_curves(history)


def cmd_evaluate() -> None:
    _, val_loader, dataset = build_dataloaders()
    class_names = dataset.classes  # 以 ImageFolder 實際順序為準，避免錯位

    all_labels, all_preds, _ = evaluate(val_loader, class_names)
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_metrics_by_class(all_labels, all_preds, class_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="SwinV2 語音情緒辨識")
    parser.add_argument("command", choices=["download", "preprocess", "train",
                                            "evaluate", "all"])
    args = parser.parse_args()

    if args.command == "download":
        cmd_download()
    elif args.command == "preprocess":
        cmd_preprocess()
    elif args.command == "train":
        cmd_train()
    elif args.command == "evaluate":
        cmd_evaluate()
    elif args.command == "all":
        cmd_download()
        cmd_preprocess()
        cmd_train()
        cmd_evaluate()


if __name__ == "__main__":
    main()
