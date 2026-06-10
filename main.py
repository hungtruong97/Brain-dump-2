import json
import os
import uuid
from datetime import datetime, timezone

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google.cloud import firestore
from pydantic import BaseModel
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

app = FastAPI(title="Brain Dump")

_db: firestore.Client | None = None
_embedding_model: TextEmbeddingModel | None = None
_llm: GenerativeModel | None = None
_vertex_initialized = False

NIGHTLY_PROMPT = """You are processing a person's raw daily brain dump.
Input is a list of timestamped notes from today.

Do the following:
1. Split each note into atomic thoughts (one idea per unit)
2. Detect topics for each atomic thought (2-3 word slugs)
3. Group all atomic thoughts into emergent categories
4. Write a one paragraph honest day summary
5. Extract all action items
6. Read the mood/energy tone of the day
7. Compare topics against existing threads (provided below)
   and return which existing threads these notes belong to,
   and which new threads should be created

Return ONLY valid JSON, no markdown, no preamble:
{{
  "atomic_notes": [
    {{
      "note_id": "<original note uuid>",
      "thoughts": ["atomic thought 1", "atomic thought 2"],
      "topics": ["topic-slug-one", "topic-slug-two"]
    }}
  ],
  "categories": {{
    "CategoryName": ["note_id_1", "note_id_2"]
  }},
  "summary": "...",
  "action_items": ["action 1", "action 2"],
  "mood_signal": "...",
  "thread_matches": [
    {{"thread_id": "existing-slug", "note_ids": ["note_id_1"]}}
  ],
  "new_threads": [
    {{"topic": "Human Readable Name", "topic_slug": "human-readable-name", "note_ids": ["note_id_1"]}}
  ]
}}

Existing threads:
{threads_json}

Today's notes:
{notes_json}"""


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        project = os.getenv("GCP_PROJECT_ID")
        database = os.getenv("FIRESTORE_DATABASE", "(default)")
        _db = firestore.Client(project=project, database=database)
    return _db


def init_vertex():
    global _vertex_initialized
    if not _vertex_initialized:
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"),
            location=os.getenv("VERTEX_REGION", "us-central1"),
        )
        _vertex_initialized = True


def get_embedding_model() -> TextEmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        init_vertex()
        _embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return _embedding_model


def get_llm() -> GenerativeModel:
    global _llm
    if _llm is None:
        init_vertex()
        _llm = GenerativeModel("gemini-2.0-flash")
    return _llm


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


@app.post("/process", status_code=200)
async def process_notes():
    db = get_db()
    today = datetime.now(timezone.utc).date().isoformat()
    today_start = datetime(
        *[int(p) for p in today.split("-")], tzinfo=timezone.utc
    )

    all_unprocessed = (
        db.collection("notes")
        .where("processed", "==", False)
        .get()
    )
    raw_notes = [
        n for n in all_unprocessed
        if n.get("timestamp") and n.get("timestamp").replace(tzinfo=timezone.utc) >= today_start
    ]

    if not raw_notes:
        return {"status": "no unprocessed notes for today"}

    notes_data = [
        {
            "id": n.id,
            "text": n.get("raw_text"),
            "timestamp": n.get("timestamp").isoformat(),
        }
        for n in raw_notes
    ]

    threads_data = [
        {"thread_id": t.id, "topic": t.get("topic")}
        for t in db.collection("threads").get()
    ]

    prompt = NIGHTLY_PROMPT.format(
        threads_json=json.dumps(threads_data, indent=2),
        notes_json=json.dumps(notes_data, indent=2),
    )

    response = get_llm().generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=4096,
        ),
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid JSON: {e}")

    # Save daily summary
    db.collection("daily_summary").document(today).set({
        "date": today,
        "categories": result.get("categories", {}),
        "summary": result.get("summary", ""),
        "action_items": result.get("action_items", []),
        "mood_signal": result.get("mood_signal", ""),
        "patterns": [],
    })

    # Create new threads
    now = datetime.now(timezone.utc)
    for thread in result.get("new_threads", []):
        slug = thread.get("topic_slug", "").strip()
        if not slug:
            continue
        db.collection("threads").document(slug).set({
            "topic": thread.get("topic", slug),
            "note_ids": thread.get("note_ids", []),
            "first_seen": now,
            "last_seen": now,
            "claude_summary": "",
        })

    # Update existing threads
    for match in result.get("thread_matches", []):
        thread_id = match.get("thread_id", "").strip()
        note_ids = match.get("note_ids", [])
        if thread_id and note_ids:
            db.collection("threads").document(thread_id).update({
                "note_ids": firestore.ArrayUnion(note_ids),
                "last_seen": now,
            })

    # Update individual notes with atomic thoughts + topics, mark processed
    atomic_map = {a["note_id"]: a for a in result.get("atomic_notes", [])}
    for note in raw_notes:
        update = {"processed": True}
        if note.id in atomic_map:
            update["atomic_notes"] = atomic_map[note.id].get("thoughts", [])
            update["topics"] = atomic_map[note.id].get("topics", [])
        db.collection("notes").document(note.id).update(update)

    return {
        "status": "processed",
        "notes_count": len(raw_notes),
        "date": today,
        "new_threads": len(result.get("new_threads", [])),
        "thread_matches": len(result.get("thread_matches", [])),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
