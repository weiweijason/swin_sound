"""建立 ImageFolder 資料集與 DataLoader。

注意：ImageFolder 會依「資料夾名稱字母順序」決定類別索引，
因此這裡統一用 dataset.classes 作為類別名稱來源，
避免手動寫死順序造成標籤錯位。
"""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .. import config


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def build_dataloaders(batch_size: int = config.BATCH_SIZE,
                      train_ratio: float = config.TRAIN_RATIO,
                      seed: int = 42,
                      num_workers: int = 2):
    """回傳 (train_loader, val_loader, dataset)。"""
    dataset = datasets.ImageFolder(root=str(config.MEL_DIR),
                                   transform=build_transform())
    print(f"Number of classes: {len(dataset.classes)} ({dataset.classes})")

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, dataset
