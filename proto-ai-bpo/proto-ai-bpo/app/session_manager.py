"""
session_manager.py
===================
Owns the lifecycle of a single customer interaction ("session"):

  1. create_session()  -> new session_id, in-memory record starts
  2. log_turn()         -> every customer/agent exchange appended
  3. close_session()     -> summary generated, synced as one row to the
                            Ledger Google Sheet (Sheet 3)
  4. generate_pdf()      -> renders a session transcript + summary as a
                            downloadable PDF for the admin panel

The in-memory store is intentionally simple (dict) for the prototype;
swap `_sessions` for Redis/Postgres when moving past proto stage — the
public interface below would not need to change.
"""

import logging
import uuid
from datetime import datetime, timezone
from functools import lru_cache

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from . import config
from .sheets_client import get_sheets_client

logger = logging.getLogger("proto_bpo.session_manager")

LEDGER_COLUMNS = [
    "session_id", "customer_id", "customer_name", "channel", "language",
    "started_at", "ended_at", "duration_seconds", "turn_count",
    "resolution_status", "summary", "sentiment", "agent_model",
]


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    # -- lifecycle -----------------------------------------------------

    def create_session(
        self, customer_id: str = "", customer_name: str = "", channel: str = "voice",
        language: str = config.DEFAULT_LANGUAGE,
    ) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "session_id": session_id,
            "customer_id": customer_id or "unknown",
            "customer_name": customer_name or "unknown",
            "channel": channel,
            "language": language,
            "started_at": datetime.now(timezone.utc),
            "ended_at": None,
            "turns": [],
            "resolution_status": "open",
            "agent_model": config.LANGUAGE_MODEL_MAP.get(language, ""),
        }
        logger.info("Session created: %s (%s, %s)", session_id, channel, language)
        return session_id

    def log_turn(self, session_id: str, speaker: str, text: str, language: str | None = None) -> None:
        session = self._sessions.get(session_id)
        if not session:
            logger.warning("log_turn called for unknown session %s", session_id)
            return
        session["turns"].append(
            {
                "speaker": speaker,  # "customer" | "agent"
                "text": text,
                "language": language or session["language"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_locked_language(self, session_id: str) -> str | None:
        """The language this session has settled on, once the customer has
        spoken at least once. Returns None before the first utterance, so
        the caller knows to run free auto-detect on turn one."""
        session = self._sessions.get(session_id)
        return session.get("locked_language") if session else None

    def set_locked_language(self, session_id: str, language: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session["locked_language"] = language
            session["language"] = language  # keep ledger's recorded language current too

    def close_session(
        self, session_id: str, resolution_status: str = "resolved",
        sentiment: str = "neutral", summary: str | None = None,
    ) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Unknown session_id {session_id}")

        session["ended_at"] = datetime.now(timezone.utc)
        session["resolution_status"] = resolution_status
        session["sentiment"] = sentiment
        session["summary"] = summary or self._auto_summary(session)

        duration = (session["ended_at"] - session["started_at"]).total_seconds()

        row = [
            session["session_id"],
            session["customer_id"],
            session["customer_name"],
            session["channel"],
            session["language"],
            session["started_at"].isoformat(),
            session["ended_at"].isoformat(),
            round(duration, 1),
            len(session["turns"]),
            session["resolution_status"],
            session["summary"],
            session["sentiment"],
            session["agent_model"],
        ]

        synced = get_sheets_client().append_ledger_entry(row)
        if not synced:
            logger.warning(
                "Session %s closed locally but failed to sync to ledger sheet", session_id
            )

        logger.info("Session closed: %s (%.1fs, %d turns)", session_id, duration, len(session["turns"]))
        return session

    def _auto_summary(self, session: dict) -> str:
        customer_lines = [t["text"] for t in session["turns"] if t["speaker"] == "customer"]
        if not customer_lines:
            return "No customer input recorded."
        preview = " / ".join(customer_lines[:3])
        return f"Customer raised: {preview[:280]}"

    # -- read access for the admin panel --------------------------------

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def list_active_sessions(self) -> list[dict]:
        return [s for s in self._sessions.values() if s["ended_at"] is None]

    def list_all_local_sessions(self) -> list[dict]:
        return list(self._sessions.values())

    def get_ledger_from_sheet(self) -> list[dict]:
        """Authoritative history — reads directly from Sheet 3."""
        return get_sheets_client().get_ledger()

    # -- PDF export -------------------------------------------------------

    def generate_pdf(self, session_id: str, out_path: str) -> str:
        session = self._sessions.get(session_id)
        if not session:
            raise KeyError(f"Unknown session_id {session_id}")

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleDark", parent=styles["Title"], textColor=colors.HexColor("#131720")
        )
        doc = SimpleDocTemplate(out_path, pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
        story = []

        story.append(Paragraph(f"{config.APP_NAME} — Session Report", title_style))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(f"Session ID: {session['session_id']}", styles["Normal"]))
        story.append(Paragraph(f"Customer: {session['customer_name']} ({session['customer_id']})", styles["Normal"]))
        story.append(Paragraph(f"Channel / Language: {session['channel']} / {session['language']}", styles["Normal"]))
        story.append(Paragraph(f"Started: {session['started_at']}", styles["Normal"]))
        story.append(Paragraph(f"Ended: {session.get('ended_at', 'in progress')}", styles["Normal"]))
        story.append(Paragraph(f"Status: {session.get('resolution_status', 'open')}", styles["Normal"]))
        story.append(Paragraph(f"Sentiment: {session.get('sentiment', 'n/a')}", styles["Normal"]))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(f"Summary: {session.get('summary', '')}", styles["Normal"]))
        story.append(Spacer(1, 0.6 * cm))

        table_data = [["#", "Speaker", "Text", "Timestamp"]]
        for i, turn in enumerate(session["turns"], start=1):
            table_data.append([str(i), turn["speaker"], turn["text"], turn["timestamp"]])

        table = Table(table_data, colWidths=[1 * cm, 2.5 * cm, 9 * cm, 4 * cm])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131720")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ]
            )
        )
        story.append(table)
        doc.build(story)
        return out_path


@lru_cache(maxsize=1)
def get_session_manager() -> SessionManager:
    return SessionManager()