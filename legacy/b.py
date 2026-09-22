import os
import glob
import librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 🟢 救星 1：強制關閉互動顯示後台，極大節省 RAM
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc              # 🟢 救星 2：引入系統垃圾回收機制

target_emotions = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]

output_dir = '/content/dataset'
for emo in target_emotions:
    os.makedirs(os.path.join(output_dir, emo), exist_ok=True)

def save_melspec(file_path, emotion_label, prefix):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=3.0)

        if len(y) < 3.0 * sr:
            y = np.pad(y, (0, int(3.0 * sr) - len(y)), 'constant')

        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)

        fig = plt.figure(figsize=(3, 3))
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.imshow(melspec_db, cmap='viridis', aspect='auto', origin='lower')

        filename = f"{prefix}_{os.path.basename(file_path).replace('.wav', '.jpg')}"
        save_path = os.path.join(output_dir, emotion_label, filename)

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # 🟢 救星 3：徹底清除當前畫布的所有物件
        fig.clf()
        plt.close('all')

    except Exception as e:
        pass

# ==========================================
# 定義所有資料集的對照表與路徑
# ==========================================
ravdess_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
tess_map = {
    "neutral": "neutral", "happy": "happy", "sad": "sad",
    "angry": "angry", "fear": "fearful", "disgust": "disgust",
    "surprise": "surprised"
}
crema_map = {
    "NEU": "neutral", "HAP": "happy", "SAD": "sad",
    "ANG": "angry", "FEA": "fearful", "DIS": "disgust"
}

ravdess_files = glob.glob("/content/ravdess_audio/**/*.wav", recursive=True)
tess_files = glob.glob("/content/tess_audio/**/*.wav", recursive=True)
crema_files = glob.glob("/content/cremad_audio/**/*.wav", recursive=True)

# 把所有檔案與對應的標籤、前綴整理成一個大清單
all_tasks = []

for f in ravdess_files:
    code = os.path.basename(f).split("-")[2]
    if code in ravdess_map:
        all_tasks.append((f, ravdess_map[code], "RAV"))

for f in tess_files:
    fname = os.path.basename(f).lower()
    for key, val in tess_map.items():
        if key in fname:
            all_tasks.append((f, val, "TESS"))
            break

for f in crema_files:
    parts = os.path.basename(f).split("_")
    if len(parts) >= 3:
        code = parts[2]
        if code in crema_map:
            all_tasks.append((f, crema_map[code], "CREMA"))

print(f"總共準備處理 {len(all_tasks)} 筆音檔...")

# ==========================================
# 執行轉換並定期清理記憶體
# ==========================================
for i, (f, label, prefix) in enumerate(tqdm(all_tasks)):
    save_melspec(f, label, prefix)

    # 🟢 救星 4：每處理 100 張圖，強制清空一次系統 RAM 垃圾
    if i % 100 == 0:
        gc.collect()

print("\n🎉 三大資料集融合與前處理全部完成！RAM 安全下莊！")