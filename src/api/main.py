
import sys
import os
import io
import time
import uuid
import atexit
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.rag.query_engine import QueryEngine
from src.summarization.inference import summarize

# ── Constants ──────────────────────────────────────────────────────────────
MAX_UPLOAD_MB   = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_DIR      = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIZE_TIMEOUT_S = 180   # 3 min max for summarization on CPU

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NyayLens API",
    description="Production API for Legal Chat, Document QA, and Summarization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nyay-lens.vercel.app",  # Production Vercel URL
        "http://localhost:5173",         # Local Vite dev server
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup / Shutdown ─────────────────────────────────────────────────────
async def cleanup_loop():
    """Background task to remove leftover files older than 2 hours."""
    while True:
        now = time.time()
        for f in UPLOAD_DIR.glob("*"):
            if f.is_file() and (now - f.stat().st_mtime) > 7200:
                try:
                    f.unlink()
                except Exception as e:
                    print(f"Cleanup error: {e}")
        await asyncio.sleep(3600)  # Check every hour

@app.on_event("startup")
async def startup():
    global query_engine
    print("Initializing NyayLens Backend...")
    query_engine = QueryEngine()
    
    # Start the infinite cleanup loop
    asyncio.create_task(cleanup_loop())
    print("✓ Backend ready. Background cleanup active.")

@app.on_event("shutdown")
def shutdown():
    """Clean up all uploaded files on server shutdown."""
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print("✓ Uploads directory cleaned on shutdown.")

# ── Schema ─────────────────────────────────────────────────────────────────
class UnifiedRequest(BaseModel):
    message: str
    filepath: Optional[str] = None
    top_k: int = 5
    chat_history: Optional[list] = []

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 4000:
            raise ValueError("Message too long (max 4000 characters)")
        return v.strip()

# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/")
@app.get("/api/health")
def health():
    return {
        "status":  "online",
        "service": "NyayLens API",
        "version": "1.0.0",
        "models":  ["Legal-BERT", "Legal-PEGASUS", "Llama-3.1-8B (Groq)"],
        "index":   "FAISS 298K vectors",
    }

# ── Upload ─────────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts .pdf and .txt files up to 10 MB.
    PDFs are extracted to plain text via pdfplumber.
    Returns a server filepath for subsequent /api/chat calls.
    """
    import pdfplumber

    # 1. Validate extension
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    # 2. Read with size guard
    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB."
        )
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 3. Unique name to avoid collisions
    uid       = uuid.uuid4().hex[:8]
    safe_name = f"{uid}_{Path(filename).stem}"

    # 4. Extract / save
    if ext == ".pdf":
        text_parts = []
        try:
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t.strip())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

        if not text_parts:
            raise HTTPException(
                status_code=422,
                detail="PDF contains no readable text. It may be a scanned image — please use a searchable PDF."
            )

        out_path = UPLOAD_DIR / f"{safe_name}.txt"
        out_path.write_text("\n\n".join(text_parts), encoding="utf-8")
        return {"filepath": str(out_path), "filename": filename, "pages": len(text_parts), "size_kb": round(len(raw_bytes)/1024, 1)}

    else:
        out_path = UPLOAD_DIR / f"{safe_name}.txt"
        out_path.write_bytes(raw_bytes)
        return {"filepath": str(out_path), "filename": filename, "size_kb": round(len(raw_bytes)/1024, 1)}


# ── Chat ───────────────────────────────────────────────────────────────────
@app.post("/api/chat")
def chat(request: UnifiedRequest):
    """
    Unified intent-aware chat endpoint.
    Routes to: Summarization | Document QA | Global RAG
    """
    message_lower = request.message.lower()

    print(f"\n[BACKEND] '{request.message[:80]}' | file={os.path.basename(request.filepath) if request.filepath else 'None'}")

    # Validate filepath if provided
    if request.filepath:
        if not os.path.exists(request.filepath):
            return JSONResponse(
                status_code=404,
                content={"answer": "The uploaded document could not be found on the server. Please re-upload the file.", "sources": []}
            )

    try:
        # ── Route 1: Summarization (with timeout) ──────────────────────────
        if "summarize" in message_lower or "summary" in message_lower:
            if not request.filepath:
                return {
                    "answer": "Please **upload a PDF or text file** first using the 📎 button, then ask me to summarize it.",
                    "sources": []
                }
            print("[BACKEND] → Summarization pipeline")
            summary_dict = summarize(request.filepath)
            return {
                "answer": "__STRUCTURED_SUMMARY__",
                "summary": summary_dict,
                "sources": [{"judgment_id": os.path.basename(request.filepath), "score": 1.0}]
            }

        # ── Route 2: Document QA ────────────────────────────────────────────
        if request.filepath:
            print("[BACKEND] → Document QA")
            return query_engine.query_with_document(request.message, request.filepath, chat_history=request.chat_history)

        # ── Route 3: Global RAG ─────────────────────────────────────────────
        print("[BACKEND] → Global RAG")
        return query_engine.query(request.message, top_k=request.top_k, chat_history=request.chat_history)

    except Exception as e:
        print(f"[BACKEND ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


# ── Cleanup old uploads (files older than 2 hours) ─────────────────────────
@app.delete("/api/upload/{filename}")
def delete_upload(filename: str):
    """Explicit delete for a specific upload."""
    target = UPLOAD_DIR / filename
    if target.exists() and target.is_file():
        target.unlink()
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="File not found.")
