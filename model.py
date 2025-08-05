import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torchvision.models import resnet18

class MambaMultimodalModel(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, hidden_dim=512, num_classes=4):
        super().__init__()
        self.text_encoder = Mamba(d_model=text_dim, d_state=16, expand=2, dropout=0.2)
        self.image_encoder = resnet18(pretrained=True)
        self.image_encoder.fc = nn.Identity()  # Remove classifier

        self.proj_image = nn.Linear(512, hidden_dim)
        self.proj_text = nn.Linear(text_dim, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, text_seq):
        image_feat = self.image_encoder(image)
        text_feat = self.text_encoder(text_seq).mean(dim=1)  # B, L, D -> B, D

        image_proj = self.proj_image(image_feat)
        text_proj = self.proj_text(text_feat)
        fused = torch.cat([image_proj, text_proj], dim=1)
        return self.fusion(fused)
