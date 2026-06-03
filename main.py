import os
import uuid
from datetime import datetime, timezone

import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import firestore
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

app = FastAPI(title="Brain Dump")

_db: firestore.Client | None = None
_embedding_model: TextEmbeddingModel | None = None


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        project = os.getenv("GCP_PROJECT_ID")
        database = os.getenv("FIRESTORE_DATABASE", "(default)")
        _db = firestore.Client(project=project, database=database)
    return _db


def get_embedding_model() -> TextEmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("VERTEX_REGION", "us-central1"),
        )
        _embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return _embedding_model


def embed(text: str) -> list[float]:
    model = get_embedding_model()
    result = model.get_embeddings([text])
    return result[0].values


class NoteRequest(BaseModel):
    text: str


@app.post("/log", status_code=200)
async def log_note(note: NoteRequest):
    if not note.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    text = note.text.strip()
    embedding = embed(text)

    db = get_db()
    note_id = str(uuid.uuid4())
    doc = {
        "raw_text": text,
        "timestamp": datetime.now(timezone.utc),
        "processed": False,
        "embedding": embedding,
        "atomic_notes": [],
        "topics": [],
    }
    db.collection("notes").document(note_id).set(doc)
    return {"id": note_id, "status": "saved"}


@app.get("/health")
async def health():
    return {"status": "ok"}
