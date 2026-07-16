"""
voice_engine.py
================
Handles the "advanced voice operation" side of the system:
  - Speech-to-Text (STT) via faster-whisper (runs fully local/offline)
  - Language detection (from STT output, reused to pick the Ollama model
    and the RAG persona/company context)
  - Text-to-Speech (TTS) via edge-tts (online, natural multilingual
    voices) with a pyttsx3 offline fallback for fully air-gapped setups.

This module is intentionally decoupled from the web framework — it can be
called from a WebSocket handler (streaming calls), a REST endpoint (batch
audio upload), or a telephony bridge (e.g. Asterisk/Twilio Media Streams)
without modification.
"""

import io
import logging
import asyncio
from functools import lru_cache

from faster_whisper import WhisperModel

from . import config

logger = logging.getLogger("proto_bpo.voice_engine")

# faster-whisper language codes -> our internal language keys
WHISPER_LANG_TO_INTERNAL = {
    "en": "english", "hi": "hindi", "ta": "tamil", "ml": "malayalam",
    "te": "telugu", "kn": "kannada", "bn": "bengali", "mr": "marathi",
    "gu": "gujarati", "pa": "punjabi", "ur": "urdu", "es": "spanish",
    "fr": "french", "ar": "arabic",
}
# ... and the reverse, used to force-decode once we already know the
# customer's language (skips the more error-prone auto-detect pass).
INTERNAL_TO_WHISPER_LANG = {v: k for k, v in WHISPER_LANG_TO_INTERNAL.items()}

# Confidence needed to override an already-locked language mid-session.
# Keeps the conversation stable (no flip-flopping on a noisy word or two)
# while still letting a customer genuinely switch languages.
LANGUAGE_SWITCH_CONFIDENCE_THRESHOLD = 0.85


class VoiceEngine:
    def __init__(self):
        logger.info(
            "Loading faster-whisper STT model '%s' on %s",
            config.STT_MODEL_SIZE, config.STT_DEVICE,
        )
        self._stt_model = WhisperModel(
            config.STT_MODEL_SIZE,
            device=config.STT_DEVICE,
            compute_type="int8" if config.STT_DEVICE == "cpu" else "float16",
        )

    # -- Speech to Text ---------------------------------------------------

    def transcribe(self, audio_path: str, hint_language: str | None = None) -> dict:
        """
        Transcribe an audio file (wav/mp3/ogg) to text and detect language.

        On the customer's FIRST utterance, `hint_language` should be None so
        Whisper freely auto-detects the language from their opening words.

        On every later turn in the same session, pass the language already
        locked for that session as `hint_language`. This forces Whisper to
        decode directly in that language (skipping the auto-detect guess),
        which is both faster and noticeably more accurate for mid/low
        resource languages like Tamil, Malayalam, etc. — auto-detect from a
        short audio clip is the least reliable part of Whisper, so we only
        pay that cost once per session, not every turn.

        If the customer's actual speech doesn't match the hint at all
        (very low confidence against the forced language), we fall back to
        one auto-detect pass so a genuine language switch isn't missed.

        Returns {"text": str, "language": internal_language_key, "confidence": float}
        """
        whisper_lang = INTERNAL_TO_WHISPER_LANG.get(hint_language) if hint_language else None

        segments, info = self._stt_model.transcribe(
            audio_path, beam_size=5, language=whisper_lang
        )
        text = " ".join(seg.text.strip() for seg in segments)
        confidence = float(info.language_probability)

        # Forced-language decode with very low confidence usually means the
        # customer switched languages — redo once with free auto-detect.
        if whisper_lang and confidence < 0.4:
            logger.info(
                "Forced decode as '%s' had low confidence (%.2f); re-running with auto-detect.",
                hint_language, confidence,
            )
            segments, info = self._stt_model.transcribe(audio_path, beam_size=5, language=None)
            text = " ".join(seg.text.strip() for seg in segments)
            confidence = float(info.language_probability)

        internal_lang = WHISPER_LANG_TO_INTERNAL.get(info.language, config.DEFAULT_LANGUAGE)
        return {
            "text": text.strip(),
            "language": internal_lang,
            "confidence": round(confidence, 3),
        }

    # -- Text to Speech ----------------------------------------------------

    async def synthesize(self, text: str, language: str, out_path: str) -> str:
        """
        Renders `text` to speech at `out_path` (mp3) using the voice mapped
        to `language`. Returns the path written.

        If the configured engine is edge-tts (cloud) and the call fails —
        e.g. due to a network/firewall issue — falls back automatically to
        the offline pyttsx3 engine so a single flaky connection doesn't
        take down the whole voice turn.
        """
        voice = config.LANGUAGE_VOICE_MAP.get(
            language, config.LANGUAGE_VOICE_MAP[config.DEFAULT_LANGUAGE]
        )

        if config.TTS_ENGINE == "edge-tts":
            try:
                import edge_tts

                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(out_path)
                return out_path
            except Exception:
                logger.exception(
                    "edge-tts synthesis failed (likely a network issue) — "
                    "falling back to offline pyttsx3 for this turn."
                )
                # fall through to offline engine below

        # Offline fallback — lower quality, no network dependency.
        import pyttsx3

        engine = pyttsx3.init()
        engine.save_to_file(text, out_path)
        engine.runAndWait()

        return out_path

    def synthesize_sync(self, text: str, language: str, out_path: str) -> str:
        return asyncio.run(self.synthesize(text, language, out_path))


@lru_cache(maxsize=1)
def get_voice_engine() -> VoiceEngine:
    return VoiceEngine()