"""
model_manager.py
=================
Switches between local Ollama LLMs based on the customer's detected
language (ollama-tamil, ollama-malayalam, ollama-hindi, ...).

Design notes:
  - Each language model is lazily instantiated and then cached, so the
    first request in a language pays the connection-setup cost and every
    later turn is fast.
  - `generate()` is the single entry point the orchestrator calls; it
    accepts the language key, the user message, and an optional RAG
    context block + persona/system prompt, and returns the model's reply.
  - Falls back to DEFAULT_LANGUAGE model if a language has no dedicated
    Ollama tag configured or the tag isn't pulled locally yet.
"""

import logging
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from . import config

logger = logging.getLogger("proto_bpo.model_manager")


class ModelManager:
    def __init__(self):
        self._model_cache: dict[str, ChatOllama] = {}
        self._history: dict[str, list] = {}  # session_id -> chat turns

    # -- model resolution -----------------------------------------------

    def _resolve_model_tag(self, language: str) -> str:
        language = (language or config.DEFAULT_LANGUAGE).lower()
        tag = config.LANGUAGE_MODEL_MAP.get(language)
        if not tag:
            logger.warning(
                "No Ollama model mapped for language '%s'; falling back to default.",
                language,
            )
            tag = config.LANGUAGE_MODEL_MAP[config.DEFAULT_LANGUAGE]
        return tag

    def _get_llm(self, language: str) -> ChatOllama:
        tag = self._resolve_model_tag(language)
        if tag not in self._model_cache:
            logger.info("Loading Ollama model '%s' for language '%s'", tag, language)
            self._model_cache[tag] = ChatOllama(
                model=tag,
                base_url=config.OLLAMA_BASE_URL,
                temperature=0.4,
            )
        return self._model_cache[tag]

    # -- generation --------------------------------------------------------

    def build_system_prompt(self, persona_text: str, company_text: str) -> str:
        return (
            "You are a professional customer-service voice agent for the "
            "company described below. Follow the persona/behavior rules "
            "exactly. Keep replies concise, natural for text-to-speech, "
            "and never invent company facts that are not in the provided "
            "context.\n\n"
            f"### Persona / Behavior Rules\n{persona_text}\n\n"
            f"### Company Knowledge (retrieved context)\n{company_text}\n"
        )

    def generate(
        self,
        session_id: str,
        language: str,
        user_message: str,
        persona_text: str = "",
        company_context: str = "",
    ) -> str:
        llm = self._get_llm(language)

        turns = self._history.setdefault(session_id, [])
        if not turns:
            turns.append(
                SystemMessage(content=self.build_system_prompt(persona_text, company_text=company_context))
            )
        else:
            # keep persona/system message fresh with latest retrieved context
            turns[0] = SystemMessage(
                content=self.build_system_prompt(persona_text, company_text=company_context)
            )

        turns.append(HumanMessage(content=user_message))

        try:
            response = llm.invoke(turns)
            reply_text = response.content
        except Exception:
            logger.exception(
                "Generation failed for language '%s' (session %s)", language, session_id
            )
            reply_text = (
                "I'm sorry, I'm having trouble reaching my knowledge system right "
                "now. Could you please repeat that, or hold while I reconnect?"
            )

        turns.append(AIMessage(content=reply_text))
        return reply_text

    def reset_session(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    def available_languages(self) -> list[str]:
        return list(config.LANGUAGE_MODEL_MAP.keys())

    def loaded_models(self) -> dict[str, str]:
        """language -> model tag currently warm in memory (for admin panel status)."""
        loaded_tags = set(self._model_cache.keys())
        return {
            lang: tag for lang, tag in config.LANGUAGE_MODEL_MAP.items() if tag in loaded_tags
        }


@lru_cache(maxsize=1)
def get_model_manager() -> ModelManager:
    return ModelManager()
