import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

from backend.model.model import get_model
from backend.model.train import CastingDataset

ROOT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 64


def evaluate():
    print("Quality Vision — Avaliação no Conjunto de Teste")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        print("ERRO: Modelo não encontrado. Execute train.py primeiro.")
        return

    model = get_model(freeze_backbone=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Modelo carregado: época {checkpoint['epoch']} | Val_Acc: {checkpoint['val_acc']*100:.2f}%")

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_dataset = CastingDataset("test", test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Imagens de teste: {len(test_dataset)}")

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().squeeze().tolist()
            preds = (torch.sigmoid(logits) >= 0.5).cpu().squeeze().tolist()
            all_probs.extend(probs if isinstance(probs, list) else [probs])
            all_preds.extend(preds if isinstance(preds, list) else [preds])
            all_labels.extend(labels.tolist())

    all_preds = [int(p) for p in all_preds]
    all_labels = [int(l) for l in all_labels]

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"\nAcurácia:  {acc*100:.2f}%")
    print(f"Precisão:  {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"\nMatriz de Confusão:")
    print(f"  [[TN={cm[0][0]}, FP={cm[0][1]}],")
    print(f"   [FN={cm[1][0]}, TP={cm[1][1]}]]")
    print("\n" + classification_report(all_labels, all_preds, target_names=["OK", "Defeito"]))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "matrix": cm,
            "labels": ["OK", "Defeito"],
        },
        "business_impact": {
            "true_negatives": cm[0][0],
            "false_positives": cm[0][1],
            "false_negatives": cm[1][0],
            "true_positives": cm[1][1],
            "note": "FN (defeito aprovado) é o erro mais custoso industrialmente",
        },
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "total_test_samples": len(all_labels),
    }

    with open(LOG_DIR / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Resultados salvos. Próximo passo: suba a API e o frontend.")


if __name__ == "__main__":
    evaluate()