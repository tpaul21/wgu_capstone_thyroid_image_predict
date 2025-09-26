# File: project/model.py  (REPLACE)
import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(binary_out: bool = True) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, 1 if binary_out else 2)
    return m

class MultiModalResNet(nn.Module):
    """
    Image backbone: ResNet-18 up to pooled features.
    Tabular branch: MLP -> embedding.
    Fusion: concat(image_feat, tab_feat) -> classifier.
    """
    def __init__(self, tab_dim: int, hidden=(128, 64), dropout=0.2):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = base.fc.in_features
        self.backbone = base
        self.backbone.fc = nn.Identity()

        layers = []
        d = tab_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        self.tab_mlp = nn.Sequential(*layers) if layers else nn.Identity()
        tab_out = hidden[-1] if hidden else tab_dim

        self.head = nn.Sequential(
            nn.Linear(in_feats + tab_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),  # binary
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(x_img)  # [B, in_feats]
        tab_feat = self.tab_mlp(x_tab)   # [B, tab_out]
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.head(fused)

def build_multimodal_resnet(tab_dim: int, hidden=(128, 64), dropout=0.2) -> nn.Module:
    return MultiModalResNet(tab_dim=tab_dim, hidden=hidden, dropout=dropout)
