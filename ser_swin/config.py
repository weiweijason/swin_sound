"""全域設定：路徑、資料集、模型與訓練超參數。

所有路徑都相對於專案根目錄（本檔案的上層目錄），
因此本專案不再依賴 Google Colab 的 /content 路徑。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 路徑
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw"          # 原始 wav 存放處
RAVDESS_DIR = RAW_AUDIO_DIR / "ravdess"
TESS_DIR = RAW_AUDIO_DIR / "tess"
CREMA_DIR = RAW_AUDIO_DIR / "cremad"
MEL_DIR = DATA_DIR / "melspec"             # 前處理後的語譜圖 (jpg)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"      # 圖表輸出

# ---------------------------------------------------------------------------
# 情緒類別（統一標籤，順序即模型輸出索引 0~7）
# ---------------------------------------------------------------------------
EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised",
]
NUM_CLASSES = len(EMOTIONS)

# 各資料集原始標籤 -> 統一標籤 的對照表
RAVDESS_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
TESS_MAP = {
    "neutral": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "fear": "fearful", "disgust": "disgust",
    "surprise": "surprised",
}
CREMA_MAP = {
    "NEU": "neutral", "HAP": "happy", "SAD": "sad",
    "ANG": "angry", "FEA": "fearful", "DIS": "disgust",
}

# ---------------------------------------------------------------------------
# 音訊 / 語譜圖參數
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
CLIP_DURATION = 3.0          # 秒；不足則補零
N_MELS = 128
FMAX = 8000

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
MODEL_NAME = "swinv2_base_window8_256"
IMAGE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# 資料切分 / 訓練
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.8
BATCH_SIZE = 16
EPOCHS = 10
WEIGHT_DECAY = 0.01
# 分層學習率 (LLRD)：底層小、高層與分類頭大
LAYER_LR = [1e-6, 5e-6, 1e-5, 5e-5]
HEAD_LR = 1e-4

BEST_MODEL_FILENAME = "swinv2_ser_best.pth"
