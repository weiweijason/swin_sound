FROM python:3.11-slim

# wget：RAVDESS 下載用；ca-certificates：HTTPS 驗證
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先裝依賴，利用 Docker layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 預設只開 bash，不自動執行任何流程
CMD ["bash"]
