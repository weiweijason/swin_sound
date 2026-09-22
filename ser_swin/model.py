"""SwinV2 模型建立與權重存取。"""
from typing import Optional

import torch
import timm

from . import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(pretrained: bool = True,
                num_classes: int = config.NUM_CLASSES) -> torch.nn.Module:
    """建立 SwinV2 Base 模型。

    pretrained=False 用於評估階段：直接覆蓋已訓練權重，省去權重下載時間。
    """
    model = timm.create_model(config.MODEL_NAME, pretrained=pretrained,
                              num_classes=num_classes)
    return model


def best_model_path() -> str:
    return str(config.CHECKPOINT_DIR / config.BEST_MODEL_FILENAME)


def save_checkpoint(model: torch.nn.Module, path: Optional[str] = None) -> str:
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = path or best_model_path()
    torch.save(model.state_dict(), path)
    return path


def load_best_model(device: torch.device,
                    num_classes: int = config.NUM_CLASSES) -> torch.nn.Module:
    """建立空模型並載入最佳權重（找不到權重時拋出明確錯誤）。"""
    import os
    path = best_model_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到模型權重：{path}\n請先執行 `python main.py train` 訓練模型。")
    model = build_model(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ 已載入最佳模型權重：{path}")
    return model
