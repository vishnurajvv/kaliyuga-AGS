# PROTO — AI‑BPO Voice Orchestration System

A prototype orchestration platform for an AI‑powered BPO: local Ollama LLMs
per language, LangChain RAG over three Google Sheets, and voice-first
customer interaction (STT → LLM+RAG → TTS), with a dark professional admin
control panel.

> **Status:** Working prototype scaffold. Wire in real Ollama models,
> a Google service account, and (for production telephony) a SIP/Twilio
> bridge to go from proto → pilot.

---

## 1. Architecture

```
                         ┌─────────────────────────┐
   Customer (voice/chat) │        main.py           │  Admin (browser)
   ───────────────────▶  │   FastAPI + Orchestrator │ ◀──────────────────
                         └────────────┬─────────────┘
                                      │
        ┌───────────────┬────────────┼────────────┬───────────────┐
        ▼               ▼            ▼            ▼               ▼
  voice_engine.py  model_manager.py rag_engine.py session_manager.py admin_panel/
  (STT / TTS,      (per-language    (LangChain +  (ledger logging,   (dark theme
   faster-whisper,  Ollama models,   Chroma RAG    PDF export,        control room
   edge-tts)        switch by lang)  over Sheets    Sheet 3 sync)      UI)
                                      1 & 2)
                                      │
                                      ▼
                              sheets_client.py
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
              Sheet 1: Persona   Sheet 2: Company   Sheet 3: Ledger
              (behavior/         (details, FAQs,    (session/customer
               character)         policies)          history, read+write)
```

**One customer turn, end to end:**
1. Audio (or text) comes in → `voice_engine.transcribe()` runs STT and
   detects the spoken language.
2. `rag_engine.retrieve()` pulls the most relevant persona rows (Sheet 1)
   and company-knowledge rows (Sheet 2) for that message.
3. `model_manager.generate()` picks the Ollama model mapped to the
   detected language (e.g. `ollama-tamil`) and generates a reply grounded
   in the retrieved context.
4. `session_manager.log_turn()` records the exchange in memory.
5. `voice_engine.synthesize()` renders the reply back to speech in the
   same language.
6. On call end, `session_manager.close_session()` writes one summary row
   to the Ledger sheet (Sheet 3) and the transcript can be downloaded as
   a PDF from the admin panel.

---

## 2. File structure

```
proto-ai-bpo/
├── app/
│   ├── main.py              # 1. Orchestrator + FastAPI app (entry point)
│   ├── admin_panel/
│   │   └── index.html       # 2. Dark-theme admin control panel (frontend)
│   ├── model_manager.py     # 3. Switches Ollama model per language
│   ├── rag_engine.py        # 4. LangChain RAG over persona/company sheets
│   ├── session_manager.py   # 5. Session/customer ledger + PDF report export
│   ├── sheets_client.py     #    Shared Google Sheets connector
│   ├── voice_engine.py      #    STT (faster-whisper) + TTS (edge-tts)
│   ├── config.py            #    All configuration in one place
│   ├── credentials/         #    (you create) service_account.json goes here
│   ├── data/vector_store/   #    Chroma persistence (auto-created)
│   ├── logs/                #    (auto-created)
│   └── sessions_pdf/        #    Generated session PDFs (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Setup

### 3.1 Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally (`ollama serve`)
- A Google Cloud service account with the Sheets + Drive APIs enabled

### 3.2 Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit values as needed
```

### 3.3 Google Sheets access
1. In Google Cloud Console, create a **service account**, enable
   **Google Sheets API** and **Google Drive API**, and download its JSON key.
2. Save the key as `app/credentials/service_account.json`
   (or point `GOOGLE_CREDENTIALS_PATH` in `.env` elsewhere).
3. Open each of the 3 spreadsheets → **Share** → paste the service
   account's `client_email` → give **Editor** access (the ledger sheet is
   written to, not just read).
4. The spreadsheet IDs are already pre-filled in `.env.example` from the
   links you shared — confirm the tab name (`SHEET_*_TAB`) matches your
   actual worksheet name (defaults to `Sheet1`).

**Recommended column headers:**

| Sheet | Suggested columns |
|---|---|
| 1. Persona | `trait`, `tone`, `rule`, `example_phrase`, `do_not` |
| 2. Company | `topic`, `detail`, `category`, `policy`, `escalation_path` |
| 3. Ledger  | `session_id`, `customer_id`, `customer_name`, `channel`, `language`, `started_at`, `ended_at`, `duration_seconds`, `turn_count`, `resolution_status`, `summary`, `sentiment`, `agent_model` |

The app writes the Ledger row automatically on `close_session()` — you
only need to create the header row once.

### 3.4 Pull / create your per-language Ollama models
```bash
ollama pull llama3.1:8b            # base English model
ollama pull nomic-embed-text       # embeddings for RAG

# Language-specific models — either pull a fine-tuned community model,
# or create one from a base model + language system prompt, e.g.:
cat > Modelfile.tamil <<'EOF'
FROM llama3.1:8b
SYSTEM "You always respond fluently and naturally in Tamil unless the customer switches language."
EOF
ollama create ollama-tamil -f Modelfile.tamil
# Repeat for ollama-malayalam, ollama-hindi, etc.
```
Update the tags in `.env` (`MODEL_TAMIL`, `MODEL_MALAYALAM`, ...) to match
whatever you name them.

### 3.5 Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000** for the admin control panel.

---

## 4. API quick reference

| Endpoint | Purpose |
|---|---|
| `POST /api/session/start` | Create a session, get `session_id` |
| `POST /api/chat` | Send a text turn, get the agent's reply |
| `POST /api/voice/turn` | Upload audio → transcript + reply text + reply audio URL |
| `POST /api/session/close` | Close a session, sync summary row to Ledger sheet |
| `GET  /api/admin/overview` | Dashboard stats (active sessions, loaded models) |
| `GET  /api/admin/ledger` | Full ledger, read live from Sheet 3 |
| `GET  /api/admin/session/{id}/pdf` | Download a session transcript as PDF |
| `GET  /api/admin/persona` / `/api/admin/company` | Inspect raw sheet rows |
| `POST /api/admin/rag/refresh` | Force-rebuild the RAG index now |

---

## 5. Roadmap from proto → production
- Swap the in-memory `SessionManager._sessions` dict for Redis/Postgres.
- Add a telephony bridge (Twilio Media Streams / Asterisk / FreeSWITCH)
  feeding `voice_engine` for real inbound/outbound calling.
- Add streaming STT/TTS (websocket) instead of upload-then-respond for
  lower latency live calls.
- Add authentication (the `ADMIN_TOKEN` in `.env` is a placeholder) in
  front of `/api/admin/*` and the control panel.
- Add per-agent-model evaluation/QA sampling from the ledger.
