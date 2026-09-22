"""下載 RAVDESS / TESS / CREMA-D 三大語音情緒資料集。

- RAVDESS：Zenodo 直接下載（免登入）。
- TESS / CREMA-D：透過 Kaggle API 下載，金鑰從環境變數
  KAGGLE_USERNAME / KAGGLE_KEY 讀取（或 ~/.kaggle/kaggle.json），
  不再把金鑰寫死在程式碼中。
"""
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from .. import config

RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"


def _run(cmd: list[str]) -> None:
    """執行外部指令，失敗時拋出明確錯誤。"""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"指令執行失敗 (exit={result.returncode}): {' '.join(cmd)}")


def _ensure_kaggle_credentials() -> None:
    """確認 Kaggle 金鑰可用（環境變數或 ~/.kaggle/kaggle.json）。"""
    if Path.home().joinpath(".kaggle", "kaggle.json").exists():
        return
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        raise EnvironmentError(
            "找不到 Kaggle 金鑰。請設定環境變數 KAGGLE_USERNAME / KAGGLE_KEY，"
            "或建立 ~/.kaggle/kaggle.json（參見 .env.example）。"
        )
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    cred_file = kaggle_dir / "kaggle.json"
    cred_file.write_text(json.dumps({"username": username, "key": key}))
    os.chmod(cred_file, 0o600)


def _download_and_unzip(url_or_dataset: str, dest_dir: Path,
                         is_kaggle: bool = False, dataset_slug: Optional[str] = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir.parent / f"{dest_dir.name}.zip"

    if is_kaggle:
        _ensure_kaggle_credentials()
        _run(["kaggle", "datasets", "download", "-d", dataset_slug,
              "-p", str(dest_dir.parent)])
        # kaggle 下載的檔名為 {slug 最後一段}.zip
        zip_path = dest_dir.parent / f"{dataset_slug.split('/')[-1]}.zip"
    else:
        _run(["wget", "-q", "-O", str(zip_path), url_or_dataset])

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink(missing_ok=True)
    print(f"✅ {dest_dir.name} 下載並解壓完成 -> {dest_dir}")


def download_ravdess() -> None:
    if (config.RAVDESS_DIR / "Audio_Speech_Actors_01-24").exists():
        print("RAVDESS 已存在，略過下載。")
        return
    _download_and_unzip(RAVDESS_URL, config.RAVDESS_DIR)


def download_tess() -> None:
    if config.TESS_DIR.exists() and any(config.TESS_DIR.rglob("*.wav")):
        print("TESS 已存在，略過下載。")
        return
    _download_and_unzip(None, config.TESS_DIR,
                        is_kaggle=True, dataset_slug="ejlok1/toronto-emotional-speech-set-tess")


def download_cremad() -> None:
    if config.CREMA_DIR.exists() and any(config.CREMA_DIR.rglob("*.wav")):
        print("CREMA-D 已存在，略過下載。")
        return
    _download_and_unzip(None, config.CREMA_DIR,
                        is_kaggle=True, dataset_slug="ejlok1/cremad")


def download_all() -> None:
    download_ravdess()
    download_tess()
    download_cremad()
