"""將三大資料集的 wav 音檔轉換為 mel 語譜圖 (jpg)。

輸出結構：data/melspec/{emotion}/{prefix}_{原檔名}.jpg
"""
import gc
import os

import librosa
import matplotlib
matplotlib.use("Agg")  # 無 GUI 環境下強制使用後端，避免記憶體洩漏
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .. import config


def _save_melspec(file_path: str, emotion_label: str, prefix: str) -> None:
    """讀取音檔 -> 3 秒 mel 語譜圖 -> 存成 jpg。"""
    try:
        y, sr = librosa.load(file_path, sr=config.SAMPLE_RATE,
                             duration=config.CLIP_DURATION)
        if len(y) < config.CLIP_DURATION * sr:
            y = np.pad(y, (0, int(config.CLIP_DURATION * sr) - len(y)), "constant")

        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=config.N_MELS, fmax=config.FMAX)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        fig = plt.figure(figsize=(3, 3))
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(melspec_db, cmap="viridis", aspect="auto", origin="lower")

        filename = f"{prefix}_{os.path.basename(file_path).replace('.wav', '.jpg')}"
        save_path = os.path.join(config.MEL_DIR, emotion_label, filename)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

        # 徹底清除畫布物件，避免長程批次處理時 RAM 持續成長
        fig.clf()
        plt.close("all")
    except Exception as e:  # 單一檔案失敗不中斷整體流程
        print(f"⚠️ 處理失敗 {file_path}: {e}")


def _collect_tasks() -> list[tuple[str, str, str]]:
    """掃描三大資料集，回傳 [(wav 路徑, 統一標籤, 前綴), ...]。"""
    import glob

    tasks: list[tuple[str, str, str]] = []

    for f in glob.glob(str(config.RAVDESS_DIR / "**" / "*.wav"), recursive=True):
        code = os.path.basename(f).split("-")[2]
        if code in config.RAVDESS_MAP:
            tasks.append((f, config.RAVDESS_MAP[code], "RAV"))

    for f in glob.glob(str(config.TESS_DIR / "**" / "*.wav"), recursive=True):
        fname = os.path.basename(f).lower()
        for key, val in config.TESS_MAP.items():
            if key in fname:
                tasks.append((f, val, "TESS"))
                break

    for f in glob.glob(str(config.CREMA_DIR / "**" / "*.wav"), recursive=True):
        parts = os.path.basename(f).split("_")
        if len(parts) >= 3 and parts[2] in config.CREMA_MAP:
            tasks.append((f, config.CREMA_MAP[parts[2]], "CREMA"))

    return tasks


def build_melspec_dataset(force: bool = False) -> int:
    """執行完整的前處理流程，回傳產出的語譜圖數量。

    force=True 時先清空舊的語譜圖目錄重新產生。
    """
    if force and config.MEL_DIR.exists():
        import shutil
        shutil.rmtree(config.MEL_DIR)
    for emo in config.EMOTIONS:
        os.makedirs(os.path.join(config.MEL_DIR, emo), exist_ok=True)

    tasks = _collect_tasks()
    print(f"總共準備處理 {len(tasks)} 筆音檔...")
    if not tasks:
        raise FileNotFoundError(
            "找不到任何 wav 檔案，請先執行 `python main.py download` 下載資料集。")

    for i, (f, label, prefix) in enumerate(tqdm(tasks, desc="Melspec")):
        _save_melspec(f, label, prefix)
        if i % 100 == 0:
            gc.collect()

    total = sum(len(files) for _, _, files in os.walk(config.MEL_DIR))
    print(f"🎉 前處理完成，共 {total} 張語譜圖 -> {config.MEL_DIR}")
    return total
