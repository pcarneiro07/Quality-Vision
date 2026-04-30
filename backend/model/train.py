import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from backend.model.model import get_model

ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

BATCH_SIZE = 32
EPOCHS = 20
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
EARLY_STOPPING_PATIENCE = 5
FINETUNE_EPOCH = 5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Reduzido de 0.18 → 0.12 para devolver ~300 imagens ao treino
VAL_RATIO = 0.12

# MixUp / CutMix: probabilidade de aplicar cada técnica por batch
MIXUP_ALPHA = 0.3       # λ ~ Beta(α, α); 0 desativa MixUp
CUTMIX_ALPHA = 0.5      # λ ~ Beta(α, α); 0 desativa CutMix
MIXUP_PROB = 0.35       # probabilidade de aplicar MixUp em cada batch
CUTMIX_PROB = 0.35      # probabilidade de aplicar CutMix em cada batch
# Obs: se ambos forem sorteados no mesmo batch, CutMix tem prioridade

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CastingDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.transform = transform
        self.samples = []

        for label, class_name in enumerate(["ok", "defect"]):
            class_dir = PROCESSED_DIR / split / class_name
            if not class_dir.exists():
                continue
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    self.samples.append((img_path, label))

        if not self.samples:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em {PROCESSED_DIR / split}. "
                "Execute python backend/etl/preprocess.py primeiro."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            # --- Geométricas ---
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),

            # Simula variação de distância/ângulo da câmera industrial
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=5,
            ),

            # Simula distorção de lente e peças fora do centro
            transforms.RandomPerspective(distortion_scale=0.25, p=0.3),

            # --- Fotométricas ---
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.15,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.05),

            # Simula desfoque de câmera/vibração da esteira
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

            # --- Tensor ---
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

            # RandomErasing: simula oclusões, sujeira, reflexos na peça.
            # Aplicado APÓS ToTensor porque opera em tensores.
            # p=0.3  → 30% das imagens recebem um patch apagado
            # scale  → o patch ocupa entre 2% e 15% da área da imagem
            # ratio  → proporção (largura/altura) do patch: entre 0.3 e 3.3
            # value  → preenche com ruído aleatório (mais realista que 0)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
        ])

    # Val/test: apenas normalização — sem augmentação
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# MixUp e CutMix — técnicas de regularização em nível de batch
# ---------------------------------------------------------------------------
# Ambas funcionam misturando dois exemplos do mesmo batch, forçando o modelo
# a aprender representações suaves em vez de memorizar exemplos específicos.
#
# MixUp: combinação linear pixel-a-pixel de duas imagens + seus rótulos.
#   img_mix = λ·img_a + (1-λ)·img_b,  label_mix = λ·label_a + (1-λ)·label_b
#
# CutMix: recorta uma região retangular de img_b e cola em img_a.
#   O rótulo é misturado proporcionalmente à área do patch colado.
#   É mais eficaz que MixUp para detecção de defeitos porque preserva
#   regiões locais intactas (defeitos costumam ser localizados).
# ---------------------------------------------------------------------------

def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = MIXUP_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica MixUp a um batch inteiro. Retorna (imagens misturadas, labels misturados)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = CUTMIX_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aplica CutMix a um batch inteiro. Retorna (imagens com patch, labels misturados)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)

    _, _, H, W = images.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)

    # Centro do patch sorteado aleatoriamente
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    # Recalcula lam com base na área real do patch (bordas podem ser clampadas)
    lam_real = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    mixed_labels = lam_real * labels + (1 - lam_real) * labels[idx]
    return mixed_images, mixed_labels


class TrainingLogger:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = log_dir / "training_log.json"
        self.db_path = log_dir / "training.db"
        self.logs = []
        self._init_db()
        with open(self.json_path, "w") as f:
            json.dump([], f)

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS epoch_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER, train_loss REAL, train_acc REAL,
                val_loss REAL, val_acc REAL, lr REAL, timestamp TEXT
            )
        """)
        conn.execute("DELETE FROM epoch_logs")
        conn.commit()
        conn.close()

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, lr: float):
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": lr,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.logs.append(entry)
        with open(self.json_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO epoch_logs (epoch, train_loss, train_acc, val_loss, val_acc, lr, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (epoch, train_loss, train_acc, val_loss, val_acc, lr, entry["timestamp"])
        )
        conn.commit()
        conn.close()


def run_epoch(model, loader, criterion, optimizer, scaler, device, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            # Aplica MixUp/CutMix apenas no treino
            if is_train:
                r = np.random.rand()
                if r < CUTMIX_PROB and CUTMIX_ALPHA > 0:
                    images, labels = cutmix_batch(images, labels)
                elif r < CUTMIX_PROB + MIXUP_PROB and MIXUP_ALPHA > 0:
                    images, labels = mixup_batch(images, labels)

            with autocast(enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train and optimizer and scaler:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif is_train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Para acurácia: usa threshold 0.5 nos logits originais
            # (MixUp/CutMix produz soft labels, mas a acurácia no treino
            # é apenas indicativa — o que importa é a val_acc com labels duros)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            hard_labels = (labels >= 0.5).float()
            correct += (preds == hard_labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss / total, correct / total


def train():
    print("Quality Vision — Treinamento")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_dataset = CastingDataset("train", get_transforms("train"))
    val_dataset = CastingDataset("val", get_transforms("val"))
    print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda")
    )

    model = get_model(freeze_backbone=True).to(device)
    params = model.count_parameters()
    print(f"Parâmetros treináveis: {params['trainable']:,} / {params['total']:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = GradScaler() if device.type == "cuda" else None

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(LOG_DIR)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == FINETUNE_EPOCH:
            print(f"\n[Época {epoch}] Ativando fine-tuning...")
            model.unfreeze_backbone(from_layer=14)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, is_train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, None, device, is_train=False)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

        print(
            f"Epoch [{epoch:02d}/{EPOCHS}] "
            f"| Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% "
            f"| Val_Loss: {val_loss:.4f} | Val_Acc: {val_acc*100:.2f}% "
            f"| LR: {current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, CHECKPOINT_DIR / "best_model.pth")
            print(f"  ✓ Melhor modelo salvo — Val_Acc: {val_acc*100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping após {epoch} épocas.")
                break

    print(f"\nConcluído. Melhor Val_Acc: {best_val_acc*100:.2f}%")
    print("Próximo passo: python -m backend.model.evaluate")


if __name__ == "__main__":
    train()