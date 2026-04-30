import torch
import torch.nn as nn
from torchvision import models


class QualityInspector(nn.Module):
    def __init__(self, freeze_backbone: bool = True, dropout_rate: float = 0.45):
        super().__init__()

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, 1),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = 14) -> None:
        for i, layer in enumerate(self.features):
            if i >= from_layer:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def get_model(freeze_backbone: bool = True) -> QualityInspector:
    return QualityInspector(freeze_backbone=freeze_backbone)


if __name__ == "__main__":
    model = get_model()
    params = model.count_parameters()
    print(f"Parâmetros totais:     {params['total']:,}")
    print(f"Parâmetros treináveis: {params['trainable']:,}")
    print(f"Parâmetros congelados: {params['frozen']:,}")
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")