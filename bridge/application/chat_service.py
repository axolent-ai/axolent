"""Chat-Service: Use-Case fuer LLM-Aufrufe.

Koordiniert: Conversation-History -> Sprach-Detection -> Prompt-Building -> Provider-Router -> Audit-Log.
Kein Telegram-Code hier, nur Business-Orchestration.

Seit Phase 1: nutzt ProviderRouter statt direkt claude_cli.
Default-Provider: Claude (Modus B, CLI-Subprozess).

Seit Phase 1 (Auto-Memory): Laedt automatisch relevante Memory-Eintraege
und fuegt sie in den System-Prompt ein bevor der LLM-Call stattfindet.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from domain.conversation import ConversationTurn, build_context_block
from domain.language import detect_language
from domain.personality import build_effective_prompt
from infrastructure.audit_log import write_audit_log
from infrastructure.conversation_storage import (
    get_history,
    get_language,
    save_turn,
    set_language,
)

if TYPE_CHECKING:
    from application.memory_service import MemoryService

log = logging.getLogger(__name__)

# Modul-Level Router-Referenz (wird von main.py injiziert)
_provider_router: Optional[Any] = None

# Modul-Level MemoryService-Referenz (wird von main.py injiziert)
_memory_service: Optional["MemoryService"] = None


def set_provider_router(router: Any) -> None:
    """Injiziert den ProviderRouter in den ChatService.

    Wird beim Start von main.py aufgerufen.
    """
    global _provider_router
    _provider_router = router
    log.info("ProviderRouter injiziert: %s", router)


def set_memory_service(service: Any) -> None:
    """Injiziert den MemoryService in den ChatService.

    Wird beim Start von main.py aufgerufen.
    Ermoeglicht Auto-Memory-Loading bei jeder Anfrage.
    """
    global _memory_service
    _memory_service = service
    log.info("MemoryService in ChatService injiziert: %s", service)


def _extract_keywords(text: str) -> list[str]:
    """Extrahiert Suchbegriffe aus einer User-Nachricht.

    Strategie: Alle Worte mit > 3 Zeichen, lowercase, dedupliziert.

    Args:
        text: User-Nachricht.

    Returns:
        Liste von Keywords (laengstes zuerst).
    """
    keywords = list({w.lower() for w in text.split() if len(w) > 3})
    # Sortiert nach Laenge absteigend (laengste = spezifischste zuerst)
    keywords.sort(key=len, reverse=True)
    return keywords


def _build_memory_context(user_id: int, query: str) -> tuple[str, int]:
    """Laedt relevante Memory-Eintraege fuer den User basierend auf Query.

    Such-Strategie: Keyword-Extraktion -> Substring-Match ueber alle Layer.
    Priorisiert laengste Keywords als Hauptsuche.

    Args:
        user_id: Telegram-User-ID.
        query: Aktuelle User-Nachricht.

    Returns:
        Tuple von (Memory-Context-String, Anzahl geladener Entries).
        Leerer String + 0 wenn keine Treffer oder kein MemoryService.
    """
    if _memory_service is None:
        return "", 0

    keywords = _extract_keywords(query)
    if not keywords:
        return "", 0

    # Primaerer Suchbegriff: laengstes Keyword (spezifischste)
    primary_query = keywords[0]

    episodic = _memory_service.recall(user_id, primary_query, layer="episodic", limit=3)
    semantic = _memory_service.recall(user_id, primary_query, layer="semantic", limit=3)
    procedural = _memory_service.recall(
        user_id, primary_query, layer="procedural", limit=2
    )

    # Wenn primaerer keine Treffer: sekundaeren probieren (falls vorhanden)
    if not (episodic or semantic or procedural) and len(keywords) > 1:
        secondary_query = keywords[1]
        episodic = _memory_service.recall(
            user_id, secondary_query, layer="episodic", limit=3
        )
        semantic = _memory_service.recall(
            user_id, secondary_query, layer="semantic", limit=3
        )
        procedural = _memory_service.recall(
            user_id, secondary_query, layer="procedural", limit=2
        )

    if not (episodic or semantic or procedural):
        return "", 0

    total_entries = len(episodic) + len(semantic) + len(procedural)

    sections: list[str] = ["[GESPEICHERTE NOTIZEN]", ""]
    sections.append(
        "Diese Eintraege wurden vom User gespeichert und sind moeglicherweise "
        "relevant fuer die aktuelle Frage. Nutze sie wenn passend, ignoriere wenn nicht relevant."
    )
    sections.append("")

    if episodic:
        sections.append("Episodic (was passiert ist):")
        for entry in episodic:
            sections.append(f"  • [{entry['id']}] {entry['content']}")
        sections.append("")

    if semantic:
        sections.append("Semantic (Fakten):")
        for entry in semantic:
            category = entry.get("category", "")
            cat_part = f" (kategorie: {category})" if category else ""
            sections.append(f"  • [{entry['id']}]{cat_part} {entry['content']}")
        sections.append("")

    if procedural:
        sections.append("Procedural (Skills):")
        for entry in procedural:
            skill = entry.get("skill_name", "")
            skill_part = f" [skill: {skill}]" if skill else ""
            sections.append(f"  • [{entry['id']}]{skill_part} {entry['content']}")
        sections.append("")

    return "\n".join(sections), total_entries


class ChatResult:
    """Ergebnis eines Chat-Aufrufs.

    Attributes:
        success: True wenn der Provider eine gueltige Antwort geliefert hat.
        response: Provider-Antwort (leer bei Fehler).
        error_message: Benutzerfreundliche Fehlermeldung (leer bei Erfolg).
        error_id: Fehler-ID fuer Debugging (leer bei Erfolg).
        duration: Dauer des Provider-Aufrufs in Sekunden.
        detected_language: Erkannte Sprache der User-Nachricht.
        provider_name: Name des verwendeten Providers.
    """

    __slots__ = (
        "success",
        "response",
        "error_message",
        "error_id",
        "duration",
        "detected_language",
        "provider_name",
    )

    def __init__(
        self,
        success: bool = False,
        response: str = "",
        error_message: str = "",
        error_id: str = "",
        duration: float = 0.0,
        detected_language: str = "de",
        provider_name: str = "claude",
    ) -> None:
        self.success = success
        self.response = response
        self.error_message = error_message
        self.error_id = error_id
        self.duration = duration
        self.detected_language = detected_language
        self.provider_name = provider_name


async def process_user_message(
    text: str,
    user_id: int | None,
    chat_id: int | None,
    username: str | None,
    system_prompt: str,
    language_override: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> ChatResult:
    """Verarbeitet eine User-Nachricht: History laden, Sprache erkennen, Provider aufrufen, Audit loggen.

    Args:
        text: User-Nachricht.
        user_id: Telegram User-ID.
        chat_id: Telegram Chat-ID.
        username: Telegram Username.
        system_prompt: Base System-Prompt (aus PersonalityLoader).
        language_override: If set, use this language instead of detecting.
        provider_name: Optionaler Provider-Name (None = Default aus Router).

    Returns:
        ChatResult mit Erfolg/Fehler-Details.
    """
    uid = user_id or 0
    cid = chat_id or 0

    audit: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "chat_id": chat_id,
        "username": username,
        "prompt_length": len(text),
    }

    try:
        # Sicherstellen dass Router injiziert wurde
        if _provider_router is None:
            raise RuntimeError(
                "ProviderRouter nicht initialisiert. "
                "set_provider_router() muss vor dem ersten Aufruf erfolgen."
            )

        # Load conversation history
        history = await get_history(uid, cid)

        # Auto-Memory-Loading: relevante Eintraege fuer aktuelle Frage laden
        memory_context, memory_entries_loaded = _build_memory_context(uid, text)

        # Build context-enriched prompt (history + current message)
        context_prompt = build_context_block(history, text)

        # Language: use sticky language if available, else detect + stick
        if language_override:
            lang = language_override
        else:
            sticky_lang = await get_language(uid, cid)
            if sticky_lang:
                lang = sticky_lang
            else:
                # First turn: detect and set sticky
                lang = detect_language(text)
                await set_language(uid, cid, lang)

        if lang != "de":
            log.info("Sprache fuer Chat: '%s' (sticky)", lang)

        # Effektiven Prompt mit Language-Override bauen
        effective_prompt = build_effective_prompt(system_prompt, lang)

        # Memory-Context in System-Prompt einfuegen (vor dem LLM-Call)
        if memory_context:
            effective_prompt = f"{effective_prompt}\n\n{memory_context}"

        # Provider-Router aufrufen (ersetzt direkten claude_cli Aufruf)
        result = await _provider_router.route(
            prompt=context_prompt,
            system_prompt=effective_prompt,
            provider_name=provider_name,
        )

        audit.update(
            {
                "provider": result.provider_name,
                "response_length": len(result.text),
                "duration_seconds": round(result.duration_seconds, 2),
                "detected_language": lang,
                "history_turns": len(history),
                "memory_entries_loaded": memory_entries_loaded,
            }
        )

        if result.error:
            error_id = uuid.uuid4().hex[:8]
            log.error(
                "Provider '%s' Fehler (error_id=%s): %s",
                result.provider_name,
                error_id,
                result.error,
            )
            audit["error"] = result.error
            audit["error_id"] = error_id
            return ChatResult(
                success=False,
                error_message=f"Anfrage konnte nicht verarbeitet werden. Fehler-ID: {error_id}",
                error_id=error_id,
                duration=result.duration_seconds,
                detected_language=lang,
                provider_name=result.provider_name,
            )

        response = result.text.strip()
        if not response:
            return ChatResult(
                success=False,
                error_message="Provider hat keine Antwort geliefert (leerer Output).",
                duration=result.duration_seconds,
                detected_language=lang,
                provider_name=result.provider_name,
            )

        # Save both turns to history (user + assistant)
        user_turn = ConversationTurn(role="user", content=text)
        assistant_turn = ConversationTurn(role="assistant", content=response)
        await save_turn(uid, cid, user_turn)
        await save_turn(uid, cid, assistant_turn)

        return ChatResult(
            success=True,
            response=response,
            duration=result.duration_seconds,
            detected_language=lang,
            provider_name=result.provider_name,
        )

    except FileNotFoundError:
        audit["error"] = "cli_not_found"
        log.error("CLI nicht gefunden. PATH pruefen.")
        return ChatResult(
            success=False,
            error_message=(
                "Fehler: LLM CLI ist nicht installiert oder nicht im PATH.\n"
                "Pruefe ob das entsprechende CLI im Terminal funktioniert."
            ),
        )

    except asyncio.TimeoutError:
        audit["error"] = "timeout"
        log.warning("Provider Timeout")
        return ChatResult(
            success=False,
            error_message=(
                "Provider hat zu lange gebraucht. Versuche eine kuerzere Frage."
            ),
        )

    except ValueError as e:
        # Provider nicht registriert
        audit["error"] = f"provider_error: {e}"
        log.error("Provider-Fehler: %s", e)
        return ChatResult(
            success=False,
            error_message=f"Provider-Fehler: {e}",
        )

    except RuntimeError as e:
        # Provider nicht verfuegbar oder Router nicht initialisiert
        audit["error"] = f"runtime_error: {e}"
        log.error("Runtime-Fehler: %s", e)
        return ChatResult(
            success=False,
            error_message=f"System-Fehler: {e}",
        )

    except Exception as e:
        error_id = uuid.uuid4().hex[:8]
        log.exception("Unbekannter Fehler bei der Verarbeitung (error_id=%s)", error_id)
        audit["error"] = str(e)
        audit["error_id"] = error_id
        return ChatResult(
            success=False,
            error_message=f"Ein interner Fehler ist aufgetreten. Fehler-ID: {error_id}",
            error_id=error_id,
        )

    finally:
        write_audit_log(audit)
