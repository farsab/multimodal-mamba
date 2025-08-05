import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random

class SyntheticMultimodalDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=128, image_size=224, num_classes=4):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, self.image_size, self.image_size)
        text = torch.randn(self.seq_len, 768)
        label = random.randint(0, self.num_classes - 1)
        return image, text, torch.tensor(label, dtype=torch.long)
