"""HTTP API and web UI over client.CipherLanguageClient.

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Endpoints (interactive docs at /docs):
    GET  /api/health    liveness; only answers once the models are loaded
    GET  /api/meta      cipher ids, languages, input limit, accuracy by length
    POST /api/predict   {"text": ..., "top_k": 3, "cipher": null, "chunk": true}
    GET  /              the web UI (server/static/)

Environment:
    MODELS_DIR  default "models"
    MAX_CHARS   default 100000 -- longest accepted input. Text past 10,000
                characters is predicted in 10,000-character chunks, each a full
                feature extraction, so this bounds per-request CPU time.

Run a single worker: each worker process holds its own copy of the ~720MB
of models.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from client import CipherLanguageClient, MAX_TRAINED_WINDOW
from pipeline.langcodes import LANGUAGE_NAMES

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MAX_CHARS = int(os.environ.get("MAX_CHARS", "100000"))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_state = {}


@asynccontextmanager
async def lifespan(_app):
    _state["client"] = CipherLanguageClient(models_dir=MODELS_DIR)
    yield
    _state.clear()


app = FastAPI(title="Cipher & Language Identification", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_CHARS)
    top_k: int = Field(3, ge=1, le=100)
    cipher: str | None = Field(None, description="Known cipher id: skips cipher identification")
    chunk: bool = Field(True, description=f"Average over {MAX_TRAINED_WINDOW}-character chunks "
                                          f"for longer input")


class CipherGuess(BaseModel):
    id: str
    probability: float


class LanguageGuess(BaseModel):
    code: str
    name: str
    probability: float


class PredictResponse(BaseModel):
    n_chars: int
    n_chunks: int
    word_level_applicable: bool
    cipher_source: str
    top_ciphers: list[CipherGuess]
    top_languages: list[LanguageGuess]
    family_note: str | None = None
    length_caveat: str | None = None
    elapsed_ms: float


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    client = _state["client"]
    model_eval = client.model_eval or {}
    return {
        "ciphers": client.cipher_ids,
        "languages": [{"code": c, "name": LANGUAGE_NAMES.get(c, c)} for c in client.stage_b.classes_],
        "max_chars": MAX_CHARS,
        "max_trained_window": MAX_TRAINED_WINDOW,
        "language_accuracy_by_length": model_eval.get("stage_b_language", {}).get("accuracy_by_window_size"),
        "cipher_accuracy_by_length": model_eval.get("stage_a_cipher", {}).get("accuracy_by_window_size"),
    }


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Sync handler: FastAPI runs it in its threadpool, so a long prediction
    # doesn't stall /api/health on the event loop.
    client = _state["client"]
    if not req.text.strip():
        raise HTTPException(422, "text is only whitespace")
    if req.cipher is not None and req.cipher not in client.cipher_ids:
        raise HTTPException(422, f"unknown cipher {req.cipher!r}; see /api/meta for valid ids")

    t0 = time.perf_counter()
    result = client.predict(req.text, top_k=req.top_k, known_cipher=req.cipher, chunk=req.chunk)
    return PredictResponse(
        n_chars=result["n_chars"],
        n_chunks=result["n_chunks"],
        word_level_applicable=bool(result["word_level_applicable"]),
        cipher_source=result["cipher_source"],
        top_ciphers=[CipherGuess(id=c, probability=p) for c, p in result["top_ciphers"]],
        top_languages=[LanguageGuess(code=c, name=n, probability=p) for c, n, p in result["top_languages"]],
        family_note=result.get("family_note"),
        length_caveat=result.get("length_caveat"),
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
    )


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
