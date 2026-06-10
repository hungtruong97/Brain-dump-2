import json
import math
import os
import uuid
from datetime import datetime, timezone

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
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

SEARCH_PROMPT = """You are helping a person recall their thinking on a topic.
Below are notes they wrote on different days, retrieved by semantic search.
Synthesize them into a coherent thread showing how their thinking evolved.
Be honest — if thinking is scattered or contradictory, say so.
Be concise — 3-5 sentences maximum.

Topic: {query}
Notes: {notes_json}

Return ONLY valid JSON:
{{
  "thread_summary": "...",
  "first_thought": "...",
  "latest_thought": "...",
  "evolution": "growing | stalled | contradictory | recurring"
}}"""


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
        _llm = GenerativeModel("gemini-2.5-flash")
    return _llm


def embed(text: str) -> list[float]:
    model = get_embedding_model()
    result = model.get_embeddings([text])
    return result[0].values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def llm_json(prompt: str) -> dict:
    response = get_llm().generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            max_output_tokens=4096,
        ),
    )
    return json.loads(response.text)


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

    try:
        result = llm_json(NIGHTLY_PROMPT.format(
            threads_json=json.dumps(threads_data, indent=2),
            notes_json=json.dumps(notes_data, indent=2),
        ))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid JSON: {e}")

    db.collection("daily_summary").document(today).set({
        "date": today,
        "categories": result.get("categories", {}),
        "summary": result.get("summary", ""),
        "action_items": result.get("action_items", []),
        "mood_signal": result.get("mood_signal", ""),
        "patterns": [],
    })

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

    for match in result.get("thread_matches", []):
        thread_id = match.get("thread_id", "").strip()
        note_ids = match.get("note_ids", [])
        if thread_id and note_ids:
            db.collection("threads").document(thread_id).update({
                "note_ids": firestore.ArrayUnion(note_ids),
                "last_seen": now,
            })

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


@app.get("/api/search")
async def search(q: str = Query(..., min_length=1)):
    db = get_db()

    query_embedding = embed(q)

    all_notes = db.collection("notes").get()
    scored = []
    for note in all_notes:
        note_embedding = note.get("embedding") or []
        if not note_embedding:
            continue
        score = cosine_similarity(query_embedding, note_embedding)
        scored.append((score, note))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_notes = scored[:15]

    if not top_notes:
        return {"notes": [], "thread_summary": None, "first_seen": None, "last_seen": None}

    notes_for_llm = [
        {
            "text": n.get("raw_text"),
            "timestamp": n.get("timestamp").isoformat(),
            "topics": n.get("topics") or [],
        }
        for _, n in top_notes
    ]

    try:
        synthesis = llm_json(SEARCH_PROMPT.format(
            query=q,
            notes_json=json.dumps(notes_for_llm, indent=2),
        ))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Gemini returned invalid JSON: {e}")

    timestamps = [n.get("timestamp") for _, n in top_notes if n.get("timestamp")]
    first_seen = min(timestamps).isoformat() if timestamps else None
    last_seen = max(timestamps).isoformat() if timestamps else None

    return {
        "notes": notes_for_llm,
        "thread_summary": synthesis.get("thread_summary"),
        "first_thought": synthesis.get("first_thought"),
        "latest_thought": synthesis.get("latest_thought"),
        "evolution": synthesis.get("evolution"),
        "first_seen": first_seen,
        "last_seen": last_seen,
    }


@app.get("/api/today")
async def api_today():
    db = get_db()
    today = datetime.now(timezone.utc).date().isoformat()
    doc = db.collection("daily_summary").document(today).get()
    if not doc.exists:
        return {"date": today, "exists": False}
    data = doc.to_dict()
    data["date"] = today
    data["exists"] = True
    return data


@app.get("/api/notes/today")
async def api_notes_today():
    db = get_db()
    today = datetime.now(timezone.utc).date().isoformat()
    today_start = datetime(*[int(p) for p in today.split("-")], tzinfo=timezone.utc)
    notes = []
    for n in db.collection("notes").get():
        ts = n.get("timestamp")
        if ts and ts.replace(tzinfo=timezone.utc) >= today_start:
            notes.append({
                "id": n.id,
                "raw_text": n.get("raw_text"),
                "timestamp": n.get("timestamp").isoformat(),
                "processed": n.get("processed"),
                "atomic_notes": n.get("atomic_notes") or [],
                "topics": n.get("topics") or [],
            })
    notes.sort(key=lambda x: x["timestamp"])
    return notes


@app.get("/api/threads")
async def api_threads():
    db = get_db()
    result = []
    for t in db.collection("threads").get():
        d = t.to_dict()
        d["id"] = t.id
        d["note_count"] = len(d.get("note_ids", []))
        if d.get("first_seen"):
            d["first_seen"] = d["first_seen"].isoformat()
        if d.get("last_seen"):
            d["last_seen"] = d["last_seen"].isoformat()
        d.pop("note_ids", None)
        result.append(d)
    result.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
    return result


@app.get("/api/history/{date}")
async def api_history(date: str):
    db = get_db()
    doc = db.collection("daily_summary").document(date).get()
    if not doc.exists:
        return {"date": date, "exists": False}
    data = doc.to_dict()
    data["date"] = date
    data["exists"] = True
    return data


@app.get("/")
async def dashboard():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}
