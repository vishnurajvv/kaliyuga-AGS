"""
config.py
=========
Central configuration for the PROTO AI-BPO orchestration system.

All environment-specific values (spreadsheet IDs, model names, credential
paths, service endpoints) live here so the rest of the codebase never
hard-codes them. Values are read from environment variables (.env) with
sane defaults for local development.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
SESSION_PDF_DIR = BASE_DIR / "sessions_pdf"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

for _dir in (LOG_DIR, SESSION_PDF_DIR, VECTOR_STORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# GOOGLE SHEETS — the three operational data sources
# ---------------------------------------------------------------------------
# Each spreadsheet ID is taken from the "/d/<ID>/edit" segment of the sheet URL.
GOOGLE_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CREDENTIALS_PATH", str(BASE_DIR / "credentials" / "service_account.json")
)

SPREADSHEETS = {
    # Sheet 1 — Persona / behavior / character of the service agent
    "persona": {
        "id": os.getenv("SHEET_PERSONA_ID", "1iI1IL3nOKJwbmOdSyikaq4GKrqqw6nILwkdedxObV1E"),
        "worksheet": os.getenv("SHEET_PERSONA_TAB", "Sheet1"),
    },
    # Sheet 2 — Company details (products, policies, FAQs, escalation rules)
    "company": {
        "id": os.getenv("SHEET_COMPANY_ID", "1_ZV59vmaXHQYvEcghgnZssc5A1o5hxDPp0Mu3oRohsY"),
        "worksheet": os.getenv("SHEET_COMPANY_TAB", "Sheet1"),
    },
    # Sheet 3 — Session / customer ledger (read + write)
    "ledger": {
        "id": os.getenv("SHEET_LEDGER_ID", "1br2nBLakS2e0sfSiQyQtS2n2iTHrvDdliQmd16jDVs4"),
        "worksheet": os.getenv("SHEET_LEDGER_TAB", "Sheet1"),
    },
}

# How often (seconds) the RAG engine refreshes its index from the sheets.
RAG_REFRESH_INTERVAL_SECONDS = int(os.getenv("RAG_REFRESH_INTERVAL_SECONDS", "300"))

# ---------------------------------------------------------------------------
# OLLAMA — one local model per language
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Map a language key (as detected from voice/text) to the local Ollama model
# tag that should serve it. Pull/create these tags with `ollama create` or
# `ollama pull` beforehand — see README for the Modelfile convention used
# (base model + language-specific system prompt / LoRA where available).
LANGUAGE_MODEL_MAP = {
    "english": os.getenv("MODEL_ENGLISH", "llama3.1:8b"),
    "hindi": os.getenv("MODEL_HINDI", "ollama-hindi"),
    "tamil": os.getenv("MODEL_TAMIL", "ollama-tamil"),
    "malayalam": os.getenv("MODEL_MALAYALAM", "ollama-malayalam"),
    "telugu": os.getenv("MODEL_TELUGU", "ollama-telugu"),
    "kannada": os.getenv("MODEL_KANNADA", "ollama-kannada"),
    "bengali": os.getenv("MODEL_BENGALI", "ollama-bengali"),
    "marathi": os.getenv("MODEL_MARATHI", "ollama-marathi"),
    "gujarati": os.getenv("MODEL_GUJARATI", "ollama-gujarati"),
    "punjabi": os.getenv("MODEL_PUNJABI", "ollama-punjabi"),
    "urdu": os.getenv("MODEL_URDU", "ollama-urdu"),
    "spanish": os.getenv("MODEL_SPANISH", "ollama-spanish"),
    "french": os.getenv("MODEL_FRENCH", "ollama-french"),
    "arabic": os.getenv("MODEL_ARABIC", "ollama-arabic"),
}

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "english")

# Embedding model used for the LangChain RAG vector store. Kept local via
# Ollama so no data leaves the machine.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# VOICE — speech-to-text / text-to-speech
# ---------------------------------------------------------------------------
STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "medium")  # faster-whisper model size
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")              # "cpu" or "cuda"
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge-tts")          # "edge-tts" or "pyttsx3" (offline fallback)

# Voice map: language -> TTS voice id (edge-tts naming convention)
LANGUAGE_VOICE_MAP = {
    "english": "en-IN-NeerjaNeural",
    "hindi": "hi-IN-SwaraNeural",
    "tamil": "ta-IN-PallaviNeural",
    "malayalam": "ml-IN-SobhanaNeural",
    "telugu": "te-IN-ShrutiNeural",
    "kannada": "kn-IN-SapnaNeural",
    "bengali": "bn-IN-TanishaaNeural",
    "marathi": "mr-IN-AarohiNeural",
    "gujarati": "gu-IN-DhwaniNeural",
    "punjabi": "pa-IN-GurleenNeural",
    "urdu": "ur-PK-UzmaNeural",
    "spanish": "es-ES-ElviraNeural",
    "french": "fr-FR-DeniseNeural",
    "arabic": "ar-SA-ZariyahNeural",
}

# ---------------------------------------------------------------------------
# APP / SERVER
# ---------------------------------------------------------------------------
APP_NAME = os.getenv("APP_NAME", "PROTO AI-BPO")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me-in-.env")
