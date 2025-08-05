import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import SyntheticMultimodalDataset
from model import MambaMultimodalModel
from utils import accuracy
import config as cfg

def train():
    dataset = SyntheticMultimodalDataset(num_samples=1000, seq_len=cfg.SEQ_LEN, num_classes=cfg.NUM_CLASSES)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    model = MambaMultimodalModel(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    criterion = CrossEntropyLoss()

    model.train()
    for epoch in range(cfg.EPOCHS):
        epoch_loss, epoch_acc = 0.0, 0.0
        for img, txt, label in dataloader:
            img, txt, label = img.to(cfg.DEVICE), txt.to(cfg.DEVICE), label.to(cfg.DEVICE)

            optimizer.zero_grad()
            out = model(img, txt)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            acc = accuracy(out, label)
            epoch_loss += loss.item()
            epoch_acc += acc

        scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}, Acc = {epoch_acc / len(dataloader):.4f}")

if __name__ == "__main__":
    train()
