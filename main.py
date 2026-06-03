import os
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import firestore
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Brain Dump")

_db: firestore.Client | None = None


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        project = os.getenv("GCP_PROJECT_ID")
        database = os.getenv("FIRESTORE_DATABASE", "(default)")
        _db = firestore.Client(project=project, database=database)
    return _db


class NoteRequest(BaseModel):
    text: str


@app.post("/log", status_code=200)
async def log_note(note: NoteRequest):
    if not note.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    db = get_db()
    note_id = str(uuid.uuid4())
    doc = {
        "raw_text": note.text.strip(),
        "timestamp": datetime.now(timezone.utc),
        "processed": False,
        "embedding": [],
        "atomic_notes": [],
        "topics": [],
    }
    db.collection("notes").document(note_id).set(doc)
    return {"id": note_id, "status": "saved"}


@app.get("/health")
async def health():
    return {"status": "ok"}
