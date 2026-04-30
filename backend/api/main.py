import json
import io
import base64
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.model.model import get_model

ROOT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = ROOT_DIR / "models" / "checkpoints" / "best_model.pth"
LOG_DIR = ROOT_DIR / "logs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

app = FastAPI(
    title="Quality Vision API",
    description="Sistema de inspeção automatizada de qualidade industrial",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[torch.nn.Module] = None
model_loaded = False


@app.on_event("startup")
async def load_model():
    global model, model_loaded
    if CHECKPOINT_PATH.exists():
        m = get_model(freeze_backbone=False).to(device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        m.load_state_dict(checkpoint["model_state_dict"])
        m.eval()
        model = m
        model_loaded = True
        print(f"[API] Modelo carregado — época {checkpoint['epoch']} | Val_Acc: {checkpoint['val_acc']*100:.2f}%")
    else:
        print("[API] Modelo não encontrado. Execute o treinamento primeiro.")


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.features[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(target_layer.register_forward_hook(forward_hook))
        self._hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()

        with torch.enable_grad():
            tensor = tensor.requires_grad_(True)
            logit = self.model(tensor)
            logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


def apply_heatmap(original_img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    img_np = np.array(original_img.resize((224, 224)).convert("RGB"))
    heatmap_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = (img_np * (1 - alpha) + heatmap_rgb * alpha).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    return HealthResponse(status="ok", model_loaded=model_loaded, device=str(device))


@app.post("/predict", tags=["Inspeção"])
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Formato de imagem não suportado.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob_defect = float(torch.sigmoid(model(tensor)).item())

    label = 1 if prob_defect >= 0.5 else 0
    result = "DEFEITO" if label == 1 else "APROVADA"
    confidence = prob_defect if label == 1 else (1.0 - prob_defect)

    return JSONResponse({
        "result": result,
        "label": label,
        "confidence": round(confidence, 4),
        "probability_defect": round(prob_defect, 4),
        "probability_ok": round(1.0 - prob_defect, 4),
        "filename": file.filename,
    })


@app.post("/predict-gradcam", tags=["Inspeção"])
async def predict_gradcam(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/bmp"]:
        raise HTTPException(status_code=400, detail="Formato de imagem não suportado.")

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob_defect = float(torch.sigmoid(model(tensor)).item())

    label = 1 if prob_defect >= 0.5 else 0
    result = "DEFEITO" if label == 1 else "APROVADA"
    confidence = prob_defect if label == 1 else (1.0 - prob_defect)

    gradcam = GradCAM(model)
    try:
        cam = gradcam.generate(tensor)
    finally:
        gradcam.remove_hooks()

    gradcam_b64 = apply_heatmap(img, cam, alpha=0.45)

    raw_buffer = io.BytesIO()
    Image.fromarray((cam * 255).astype(np.uint8), mode="L").save(raw_buffer, format="JPEG", quality=85)
    raw_b64 = base64.b64encode(raw_buffer.getvalue()).decode("utf-8")

    return JSONResponse({
        "result": result,
        "label": label,
        "confidence": round(confidence, 4),
        "probability_defect": round(prob_defect, 4),
        "probability_ok": round(1.0 - prob_defect, 4),
        "filename": file.filename,
        "gradcam_base64": gradcam_b64,
        "gradcam_raw_base64": raw_b64,
    })


@app.get("/metrics", tags=["Dashboard"])
async def get_training_metrics():
    json_path = LOG_DIR / "training_log.json"
    if not json_path.exists():
        return JSONResponse({"epochs": [], "status": "no_training_data"})
    with open(json_path) as f:
        logs = json.load(f)
    return JSONResponse({"epochs": logs, "status": "ok"})


@app.get("/confusion-matrix", tags=["Dashboard"])
async def get_confusion_matrix():
    results_path = LOG_DIR / "evaluation_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Execute python -m backend.model.evaluate para gerar a matriz de confusão.")
    with open(results_path) as f:
        results = json.load(f)
    return JSONResponse(results)


@app.get("/training-status", tags=["Dashboard"])
async def training_status():
    json_path = LOG_DIR / "training_log.json"
    if not json_path.exists():
        return JSONResponse({"is_training": False, "epochs_completed": 0})
    with open(json_path) as f:
        logs = json.load(f)
    if not logs:
        return JSONResponse({"is_training": False, "epochs_completed": 0})
    return JSONResponse({
        "is_training": True,
        "epochs_completed": len(logs),
        "latest_epoch": logs[-1],
    })