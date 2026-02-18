from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import open_clip
import numpy as np
import io
import os

app = FastAPI(title="Internal AI Image Matching Service")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# تحميل الموديل مرة واحدة عند بدء التشغيل
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

model.to(DEVICE)
model.eval()


# =========================
# Utilities
# =========================

def normalize_vector(v):
    return v / np.linalg.norm(v)


def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2))


# =========================
# Endpoints of the API
# =========================

@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    vector = embedding.squeeze().cpu().numpy()

    return {"embedding": vector.tolist()}


class BatchEmbeddingRequest(BaseModel):
    images: List[str]  # base64 images


@app.post("/embed-batch")
def embed_batch(req: BatchEmbeddingRequest):
    embeddings = []

    for img_base64 in req.images:
        image_data = io.BytesIO(base64.b64decode(img_base64))
        image = Image.open(image_data).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        vector = embedding.squeeze().cpu().numpy()
        embeddings.append(vector.tolist())

    return {"embeddings": embeddings}


class SimilarityRequest(BaseModel):
    vector1: List[float]
    vector2: List[float]


@app.post("/similarity")
def similarity(req: SimilarityRequest):
    v1 = np.array(req.vector1)
    v2 = np.array(req.vector2)

    sim = cosine_similarity(v1, v2)

    return {"similarity": sim}
