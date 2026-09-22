
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm

# 定義數據增強與格式轉換 (SwinV2 預設輸入通常為 256x256)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # ImageNet 標準正規化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用 ImageFolder 自動讀取資料夾分類
dataset = datasets.ImageFolder(root='/content/dataset', transform=transform)

# 切分訓練集 (80%) 與驗證集 (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 建立 DataLoader (注意 batch_size 設為 16 以免 Colab 顯存不足)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

print(f"Number of classes: {len(dataset.classes)} ({dataset.classes})")

# 載入 SwinV2 Base 模型
# num_classes 設為我們的分類數量 (4類)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('swinv2_base_window8_256', pretrained=True, num_classes=len(dataset.classes))
model = model.to(device)

print(f"Model successfully loaded on {device}!")