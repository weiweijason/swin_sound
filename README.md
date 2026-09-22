# SER_Swin — 基於 SwinV2 的語音情緒辨識

將三大公開語音情緒資料集（**RAVDESS / TESS / CREMA-D**）融合，
把音訊轉成 mel 語譜圖後，以 **SwinV2 Base** 視覺模型做 8 分類情緒辨識。

## 情緒類別（8 類）

`neutral, calm, happy, sad, angry, fearful, disgust, surprised`

> `calm` 與 `surprised` 僅 RAVDESS 提供；TESS 無 `calm`、CREMA-D 無 `calm`/`surprised`。

## 專案結構

```
sound_swin/
├── main.py                  # CLI 入口（download / preprocess / train / evaluate / all）
├── requirements.txt
├── .env.example             # Kaggle 金鑰範本（複製為 .env）
├── ser_swin/
│   ├── config.py            # 所有路徑與超參數（集中管理）
│   ├── model.py             # SwinV2 建立 / 權重存取
│   ├── train.py             # 訓練：類別加權 Loss + LLRD + 餘弦退火
│   ├── evaluate.py          # 最佳模型推論 + 分類報告
│   ├── visualize.py         # 學習曲線 / 混淆矩陣 / 指標長條圖
│   └── data/
│       ├── download.py      # 下載三大資料集（金鑰改由環境變數讀取）
│       ├── preprocess.py    # wav -> 3 秒 mel 語譜圖 (jpg)
│       └── dataset.py       # ImageFolder + DataLoader（80/20 切分）
├── data/                    # 下載與前處理產出（gitignore）
│   ├── raw/{ravdess,tess,cremad}/
│   └── melspec/{emotion}/
├── checkpoints/             # 最佳模型權重（gitignore）
├── outputs/                 # 圖表 PNG（gitignore）
└── legacy/                  # 原始 Colab 儲存格 a~h.py（僅供對照）
```

## 快速開始

```bash
# 1. 建立虛擬環境並安裝依賴
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. 設定 Kaggle 金鑰（TESS / CREMA-D 需要）
copy .env.example .env        # 再編輯 .env 填入金鑰

# 3. 依序執行各階段
python main.py download       # 下載 RAVDESS / TESS / CREMA-D
python main.py preprocess     # wav -> mel 語譜圖
python main.py train          # 訓練（預設 10 epochs）
python main.py evaluate       # 評估最佳模型 + 輸出圖表

# 或一次跑完
python main.py all
```

## 訓練策略

| 項目 | 設定 |
|---|---|
| 模型 | `swinv2_base_window8_256`（ImageNet 預訓練，256×256 輸入） |
| 資料不平衡 | 類別加權 CrossEntropy：`總樣本 / (類別數 × 類別樣本數)` |
| 微調 | 全解凍 + 分層學習率（LLRD）：底層 `1e-6` → 高層 `5e-5` → 分類頭 `1e-4` |
| 排程器 | CosineAnnealingLR |
| 優化器 | AdamW（weight_decay=0.01） |
| 切分 | 80% 訓練 / 20% 驗證（seed=42 可重現） |

## 與原始 Colab 版本的差異

1. **移除 Colab 專屬語法**：`!pip`、`!wget`、`/content/` 路徑、`drive.mount` 全部改為
   標準 Python（`subprocess` + 相對於專案根目錄的路徑）。
2. **金鑰不再硬編碼**：Kaggle 金鑰改從 `.env` / 環境變數讀取（原 a.py 中的金鑰
   已洩露於原始碼，**建議到 Kaggle 設定頁撤銷並重新產生**）。
3. **修正類別順序 bug**：原 g.py / h.py 手動寫死的 `class_names` 順序與
   `ImageFolder` 的字母排序不一致，會導致混淆矩陣標籤錯位；
   現在統一以 `dataset.classes` 為準。
4. **合併重複邏輯**：原 f.py / g.py / h.py 的評估與畫圖重疊，
   合併為 `evaluate.py` + `visualize.py`。
5. **可重現**：資料切分固定 seed；所有超參數集中在 `ser_swin/config.py`。

## 在 Colab 上跑（選用）

若仍想在 Colab 執行，把整個專案上傳後：

```python
# 第一格：掛載與安裝
from google.colab import drive
drive.mount('/content/drive')
!pip install -r requirements.txt
```

之後各階段直接 `!python main.py download` 等即可（路徑會落在專案資料夾內，
不再依賴 `/content`）。
