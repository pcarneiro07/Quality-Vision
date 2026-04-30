import os
import json
import random
import subprocess
from pathlib import Path
from typing import Tuple

from PIL import Image
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw" / "casting_data" / "casting_data"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

KAGGLE_DATASET = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"

IMG_SIZE = (224, 224)
RANDOM_SEED = 42
VAL_RATIO = 0.18

CLASS_MAP = {
    "ok_front": 0,
    "def_front": 1,
}


def log(msg: str) -> None:
    print(f"[ETL] {msg}")


def load_env() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def download_dataset() -> bool:
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        log("Dataset já encontrado — pulando download.")
        return True

    log("Dataset não encontrado. Iniciando download do Kaggle...")
    load_env()

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not username or not key:
        log("ERRO: Credenciais do Kaggle não encontradas no .env")
        return False

    try:
        import kaggle  # noqa: F401
    except ImportError:
        log("ERRO: lib 'kaggle' não instalada. Execute: pip install kaggle")
        return False

    raw_root = ROOT_DIR / "data" / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(raw_root), "--unzip"],
        capture_output=True,
        text=True,
        env=os.environ,
    )

    if result.returncode != 0:
        log(f"ERRO no download:\n{result.stderr}")
        return False

    if not RAW_DIR.exists():
        log(f"ERRO: pasta casting_data não encontrada. Conteúdo: {list(raw_root.iterdir())}")
        return False

    log("Download concluído.")
    return True


def collect_images(source_dir: Path) -> list[Tuple[Path, int]]:
    samples = []
    for class_name, label in CLASS_MAP.items():
        class_dir = source_dir / class_name
        if not class_dir.exists():
            log(f"  AVISO: {class_dir} não encontrada")
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                samples.append((img_path, label))
    return samples


def carve_val(samples: list[Tuple[Path, int]], val_ratio: float, seed: int) -> Tuple[list, list]:
    random.seed(seed)
    by_class: dict[int, list] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)

    train_out, val_out = [], []
    for items in by_class.values():
        random.shuffle(items)
        n_val = int(len(items) * val_ratio)
        val_out += items[:n_val]
        train_out += items[n_val:]

    random.shuffle(train_out)
    random.shuffle(val_out)
    return train_out, val_out


def preprocess_image(img_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)


def save_processed(
    samples: list[Tuple[Path, int]],
    split_name: str,
    output_dir: Path,
) -> list[dict]:
    split_dir = output_dir / split_name
    metadata = []

    for idx, (img_path, label) in enumerate(
        tqdm(samples, desc=f"  {split_name:10s}", unit="img")
    ):
        class_name = "ok" if label == 0 else "defect"
        dest_dir = split_dir / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            arr = preprocess_image(img_path, IMG_SIZE)

            dest_path = dest_dir / f"{split_name}_{class_name}_{idx}.jpg"

            Image.fromarray(arr).save(dest_path, quality=95)

            metadata.append({
                "file": str(dest_path.relative_to(output_dir)),
                "label": label,
                "class": class_name,
                "split": split_name,
                "original": str(img_path.name),
            })

        except Exception as e:
            log(f"  ERRO ao processar {img_path.name}: {e}")

    return metadata


def run_pipeline() -> None:
    log("Iniciando pipeline ETL — Quality Vision")

    if not download_dataset():
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_source = RAW_DIR / "train"
    test_source = RAW_DIR / "test"

    for d in [train_source, test_source]:
        if not d.exists():
            log(f"ERRO: pasta não encontrada: {d}")
            return

    train_all = collect_images(train_source)
    test_samples = collect_images(test_source)

    if not train_all or not test_samples:
        log("ERRO: nenhuma imagem encontrada.")
        return

    test_names = {p.name for p, _ in test_samples}
    train_all = [(p, l) for p, l in train_all if p.name not in test_names]

    train_samples, val_samples = carve_val(train_all, VAL_RATIO, RANDOM_SEED)

    n_ok = sum(1 for _, l in train_all + test_samples if l == 0)
    n_defect = sum(1 for _, l in train_all + test_samples if l == 1)
    log(f"Total: {len(train_all) + len(test_samples)} imagens | OK: {n_ok} | Defeito: {n_defect}")
    log(f"  Treino:    {len(train_samples)}")
    log(f"  Validação: {len(val_samples)}")
    log(f"  Teste:     {len(test_samples)}")

    all_metadata = []
    all_metadata += save_processed(train_samples, "train", PROCESSED_DIR)
    all_metadata += save_processed(val_samples, "val", PROCESSED_DIR)
    all_metadata += save_processed(test_samples, "test", PROCESSED_DIR)

    with open(PROCESSED_DIR / "metadata.json", "w") as f:
        json.dump({
            "total": len(all_metadata),
            "image_size": list(IMG_SIZE),
            "classes": {v: k for k, v in CLASS_MAP.items()},
            "splits": {
                "train": len(train_samples),
                "val": len(val_samples),
                "test": len(test_samples),
            },
            "class_distribution": {"ok": n_ok, "defect": n_defect},
            "samples": all_metadata,
        }, f, indent=2)

    log("Concluído. Próximo passo: python -m backend.model.train")


if __name__ == "__main__":
    run_pipeline()