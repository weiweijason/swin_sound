# 掛載 Google Drive (執行後會跳出授權視窗)
from google.colab import drive
import os
import json
drive.mount('/content/drive')

# 安裝視覺模型套件 timm
!pip install timm -q

# 建立資料夾並下載 RAVDESS (這部分大約需要 1-2 分鐘)
import os
os.makedirs('/content/ravdess_audio', exist_ok=True)
print("Downloading RAVDESS dataset...")
!wget -q -O ravdess.zip https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip
!unzip -q ravdess.zip -d /content/ravdess_audio
print("Download and unzip completed!")
kaggle_credentials = {
    "username": "laipao",  # 請保留引號，替換裡面的字
    "key": "845f6b7267c92a84ce78ef02d72dfac7"          # 請保留引號，替換裡面的字
}

# 2. 建立 Kaggle 預設會去抓取的隱藏資料夾
os.makedirs('/root/.kaggle', exist_ok=True)

# 3. 強制寫入實體的 kaggle.json 檔案
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(kaggle_credentials, f)

# 4. 設定嚴格的檔案權限 (Kaggle API 的安全規定)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

print("✅ Kaggle 金鑰實體檔案建立成功！")
# 1. 下載並解壓縮 TESS 資料集
print("Downloading TESS dataset...")
!kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
!unzip -q toronto-emotional-speech-set-tess.zip -d /content/tess_audio
print("TESS Ready!")

# 2. 下載並解壓縮 CREMA-D 資料集
print("Downloading CREMA-D dataset...")
!kaggle datasets download -d ejlok1/cremad
!unzip -q cremad.zip -d /content/cremad_audio
print("CREMA-D Ready!")
