"""
main.py
=======
Entry point and orchestrator for the PROTO AI-BPO system.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Responsibilities:
  - Wires together model_manager, rag_engine, voice_engine, session_manager
  - Exposes REST + WebSocket endpoints for:
        * text chat            (/api/chat)
        * voice turn (audio in -> audio out)  (/api/voice/turn)
        * admin panel data      (/api/admin/...)
        * static admin panel UI (/)
  - The `Orchestrator` class below is the single place that defines "what
    happens on one customer turn" — detect language -> retrieve persona +
    company context via RAG -> generate reply with the right Ollama model
    -> log the turn -> (voice) synthesize speech.
"""

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from . import voice_engine as voice_engine_module
from .model_manager import get_model_manager
from .rag_engine import get_rag_engine
from .voice_engine import get_voice_engine
from .session_manager import get_session_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("proto_bpo.main")


# ---------------------------------------------------------------------------
# Orchestrator — the brain that connects every module for one customer turn
# ---------------------------------------------------------------------------
class Orchestrator:
    def __init__(self):
        self.models = get_model_manager()
        self.rag = get_rag_engine()
        self.voice = get_voice_engine()
        self.sessions = get_session_manager()
        self.rag.start_auto_refresh()

    def handle_text_turn(self, session_id: str, user_text: str, language: str) -> str:
        context = self.rag.retrieve(user_text)
        self.sessions.log_turn(session_id, "customer", user_text, language)

        reply = self.models.generate(
            session_id=session_id,
            language=language,
            user_message=user_text,
            persona_text=context["persona"],
            company_context=context["company"],
        )

        self.sessions.log_turn(session_id, "agent", reply, language)
        return reply

    def handle_voice_turn(self, session_id: str, audio_path: str, out_audio_path: str) -> dict:
        locked_language = self.sessions.get_locked_language(session_id)
        stt_result = self.voice.transcribe(audio_path, hint_language=locked_language)
        language = stt_result["language"]
        user_text = stt_result["text"]

        # First utterance locks the language for the session. Later turns
        # only re-lock if Whisper is confident the customer switched
        # languages — this is what makes "detect from the first word, then
        # stay consistent" actually hold across a multi-turn conversation.
        if locked_language is None or (
            language != locked_language
            and stt_result["confidence"] >= voice_engine_module.LANGUAGE_SWITCH_CONFIDENCE_THRESHOLD
        ):
            self.sessions.set_locked_language(session_id, language)
        else:
            language = locked_language

        reply_text = self.handle_text_turn(session_id, user_text, language)

        self.voice.synthesize_sync(reply_text, language, out_audio_path)

        return {
            "transcript": user_text,
            "detected_language": language,
            "reply_text": reply_text,
            "reply_audio_path": out_audio_path,
        }


orchestrator: Orchestrator | None = None

app = FastAPI(title=config.APP_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def _startup():
    global orchestrator
    logger.info("Starting %s orchestrator...", config.APP_NAME)
    orchestrator = Orchestrator()
    logger.info("Orchestrator ready.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str | None = None
    customer_id: str = ""
    customer_name: str = ""
    language: str = config.DEFAULT_LANGUAGE
    message: str


class CloseSessionRequest(BaseModel):
    session_id: str
    resolution_status: str = "resolved"
    sentiment: str = "neutral"
    summary: str | None = None


# ---------------------------------------------------------------------------
# Chat (text) endpoints
# ---------------------------------------------------------------------------
@app.post("/api/session/start")
def start_session(customer_id: str = "", customer_name: str = "", channel: str = "chat",
                   language: str = config.DEFAULT_LANGUAGE):
    session_id = orchestrator.sessions.create_session(customer_id, customer_name, channel, language)
    return {"session_id": session_id}


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or orchestrator.sessions.create_session(
        req.customer_id, req.customer_name, "chat", req.language
    )
    reply = orchestrator.handle_text_turn(session_id, req.message, req.language)
    return {"session_id": session_id, "reply": reply}


@app.post("/api/session/close")
def close_session(req: CloseSessionRequest):
    try:
        session = orchestrator.sessions.close_session(
            req.session_id, req.resolution_status, req.sentiment, req.summary
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    orchestrator.models.reset_session(req.session_id)
    return {"status": "closed", "session_id": req.session_id, "summary": session["summary"]}


# ---------------------------------------------------------------------------
# Voice endpoints
# ---------------------------------------------------------------------------
@app.post("/api/voice/turn")
def voice_turn(session_id: str = Form(...), audio: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        # Preserve the original extension (e.g. .webm from browser MediaRecorder)
        # so the STT decoder can reliably sniff the container/codec.
        suffix = Path(audio.filename or "").suffix or ".webm"
        in_path = str(Path(tmp) / f"in_audio{suffix}")
        out_path = str(Path(tmp) / "reply.mp3")
        with open(in_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        try:
            result = orchestrator.handle_voice_turn(session_id, in_path, out_path)
        except Exception as e:
            logger.exception("Voice turn failed for session %s", session_id)
            return JSONResponse(
                status_code=500,
                content={"error": f"{type(e).__name__}: {e}", "stage": "stt_or_generation"},
            )

        final_audio_path = config.SESSION_PDF_DIR.parent / "voice_replies" / f"{session_id}_{Path(out_path).name}"
        final_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_path, final_audio_path)

    return {
        "transcript": result["transcript"],
        "detected_language": result["detected_language"],
        "reply_text": result["reply_text"],
        "reply_audio_url": f"/api/voice/audio/{final_audio_path.name}",
    }



@app.get("/api/voice/audio/{filename}")
def get_voice_audio(filename: str):
    path = config.SESSION_PDF_DIR.parent / "voice_replies" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Admin panel API
# ---------------------------------------------------------------------------
@app.get("/api/admin/overview")
def admin_overview():
    return {
        "app_name": config.APP_NAME,
        "active_sessions": len(orchestrator.sessions.list_active_sessions()),
        "total_local_sessions": len(orchestrator.sessions.list_all_local_sessions()),
        "loaded_models": orchestrator.models.loaded_models(),
        "available_languages": orchestrator.models.available_languages(),
    }


@app.get("/api/admin/sessions/active")
def admin_active_sessions():
    return orchestrator.sessions.list_active_sessions()


@app.get("/api/admin/ledger")
def admin_ledger():
    """Authoritative session/customer ledger, read live from Google Sheet 3."""
    try:
        return orchestrator.sessions.get_ledger_from_sheet()
    except Exception as e:
        logger.exception("Failed to read ledger sheet")
        return JSONResponse(status_code=502, content={"error": str(e)})


@app.get("/api/admin/session/{session_id}/pdf")
def admin_session_pdf(session_id: str):
    out_path = config.SESSION_PDF_DIR / f"{session_id}.pdf"
    try:
        orchestrator.sessions.generate_pdf(session_id, str(out_path))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")
    return FileResponse(out_path, media_type="application/pdf", filename=f"session_{session_id}.pdf")


@app.get("/api/admin/persona")
def admin_persona():
    from .sheets_client import get_sheets_client
    return get_sheets_client().get_persona_rules()


@app.get("/api/admin/company")
def admin_company():
    from .sheets_client import get_sheets_client
    return get_sheets_client().get_company_details()


@app.post("/api/admin/rag/refresh")
def admin_rag_refresh():
    orchestrator.rag.build_index()
    return {"status": "rag index rebuilt"}


# ---------------------------------------------------------------------------
# Static admin panel UI
# ---------------------------------------------------------------------------
admin_panel_dir = Path(__file__).resolve().parent / "admin_panel"
app.mount("/", StaticFiles(directory=str(admin_panel_dir), html=True), name="admin_panel")