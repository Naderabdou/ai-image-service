from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import open_clip
import numpy as np
import io
import os
import base64

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
    """
    معالجة صورة واحدة
    ✅ للـ testing والـ debugging فقط
    """
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    vector = embedding.squeeze().cpu().numpy()

    return {"embedding": vector.tolist()}


class BatchEmbeddingRequest(BaseModel):
    images: List[str]  # base64 encoded images


@app.post("/embed-batch")
def embed_batch(req: BatchEmbeddingRequest):
    """
    ✅ معالجة صور متعددة في دفعة واحدة (أسرع بكثير!)
    - بدل Job لكل صورة، نرسل الصور كلها دفعة واحدة
    - أسرع وأكفأ من single images

    Input:
    {
        "images": [
            "base64_image_1",
            "base64_image_2",
            "base64_image_3"
        ]
    }

    Output:
    {
        "embeddings": [
            [0.1, 0.2, ...],
            [0.3, 0.4, ...],
            [0.5, 0.6, ...]
        ]
    }
    """
    embeddings = []

    for i, img_base64 in enumerate(req.images):
        try:
            # Decode base64 to bytes
            try:
                image_bytes = base64.b64decode(img_base64)
            except Exception as e:
                print(f"Error decoding base64 at index {i}: {e}")
                embeddings.append(None)
                continue

            # Open image
            image_data = io.BytesIO(image_bytes)
            image = Image.open(image_data).convert("RGB")

            # Preprocess
            image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

            # Generate embedding
            with torch.no_grad():
                embedding = model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)

            # Convert to list
            vector = embedding.squeeze().cpu().numpy()
            embeddings.append(vector.tolist())

        except Exception as e:
            print(f"Error processing image at index {i}: {str(e)}")
            import traceback
            traceback.print_exc()
            embeddings.append(None)

    return {"embeddings": embeddings, "count": len([e for e in embeddings if e is not None])}


class SimilarityRequest(BaseModel):
    vector1: List[float]
    vector2: List[float]


@app.post("/similarity")
def similarity(req: SimilarityRequest):
    """
    حساب التشابه بين embedding واحد وآخر
    """
    v1 = np.array(req.vector1)
    v2 = np.array(req.vector2)

    sim = cosine_similarity(v1, v2)

    return {"similarity": sim}
