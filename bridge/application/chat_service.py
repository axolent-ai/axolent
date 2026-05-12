"""Chat-Service: Use-Case für LLM-Aufrufe.

Koordiniert: Conversation-History -> Sprach-Detection -> Prompt-Building -> Provider-Router -> Audit-Log.
Kein Telegram-Code hier, nur Business-Orchestration.

Seit Phase 1: nutzt ProviderRouter statt direkt claude_cli.
Default-Provider: Claude (Modus B, CLI-Subprozess).

Seit Phase 1 (Auto-Memory): Lädt automatisch relevante Memory-Einträge
und fügt sie in den System-Prompt ein bevor der LLM-Call stattfindet.

Seit R04: Streaming-fähig via ClaudePersistentProvider.
process_user_message_streaming() liefert einen AsyncIterator von StreamEvents
für Echtzeit-Telegram-Edits.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from application.leakage_filter import check_for_system_prompt_leakage
from domain.conversation import ConversationTurn, build_context_block
from domain.language import detect_language_with_confidence
from domain.personality import (
    SlotInfo,
    build_effective_prompt,
    build_self_awareness_block,
)
from infrastructure.audit_log import write_audit_log
from infrastructure.claude_process_pool import StreamEvent
from infrastructure.conversation_storage import (
    get_history,
    get_language,
    save_turn,
    set_language,
)
from infrastructure.conversation_storage import (
    reset_conversation as _infra_reset_conversation,
)
from infrastructure.providers.base import ProviderError

if TYPE_CHECKING:
    from application.memory_service import MemoryService
    from application.model_service import ModelService
    from application.provider_router import ProviderRouter
    from application.task_router import TaskRouter
    from infrastructure.providers.claude_persistent import ClaudePersistentProvider

log = logging.getLogger(__name__)

# Memory-Token-Budget: verhindert Prompt-Explosion bei langen /remember Einträgen
MAX_MEMORY_CHARS_PER_ENTRY = 400
MAX_MEMORY_TOTAL_CHARS = 4000  # Default-Fallback wenn Provider keine Capability meldet


def _truncate(text: str, n: int) -> str:
    """Kürzt Text auf maximal n Zeichen mit Ellipsis."""
    return text if len(text) <= n else text[: n - 3] + "..."


_STOP_WORDS_DE = frozenset(
    {
        "diese",
        "diesen",
        "dieser",
        "diesem",
        "dieses",
        "ihre",
        "ihren",
        "ihrer",
        "ihrem",
        "ihres",
        "meine",
        "meinen",
        "meiner",
        "meinem",
        "meines",
        "deine",
        "deinen",
        "seiner",
        "seinem",
        "sehr",
        "schon",
        "noch",
        "auch",
        "etwas",
        "sollte",
        "müsste",
        "könnte",
        "wollte",
        "welche",
        "welchen",
        "welcher",
        "welchem",
        "manche",
        "alles",
        "viele",
        "wenig",
        "haben",
        "machen",
        "geben",
        "nehmen",
    }
)
_STOP_WORDS_EN = frozenset(
    {
        "this",
        "that",
        "these",
        "those",
        "their",
        "there",
        "where",
        "which",
        "would",
        "could",
        "should",
        "very",
        "much",
        "more",
        "most",
        "want",
        "need",
        "have",
        "been",
        "from",
        "some",
        "many",
        "few",
        "what",
        "when",
        "while",
    }
)
_STOP_WORDS = _STOP_WORDS_DE | _STOP_WORDS_EN


def _extract_keywords(text: str) -> list[str]:
    """Extrahiert Suchbegriffe aus einer User-Nachricht.

    Strategie: Interpunktion strippen, Stop-Words filtern,
    dann alle Worte mit > 3 Zeichen, lowercase, dedupliziert.

    Args:
        text: User-Nachricht.

    Returns:
        Liste von Keywords (längstes zuerst).
    """
    # Interpunktion an Wortgrenzen entfernen (?,.:;!'"…) damit
    # "Lieblingssprache?" -> "lieblingssprache" wird
    stripped = [w.strip("?,.:;!'\"-—–…()[]{}") for w in text.split()]
    candidates = {w.lower() for w in stripped if len(w) > 3}
    keywords = list(candidates - _STOP_WORDS)
    # Sortiert nach Länge absteigend (längste = spezifischste zuerst)
    keywords.sort(key=len, reverse=True)
    return keywords


class ChatResult:
    """Ergebnis eines Chat-Aufrufs.

    Attributes:
        success: True wenn der Provider eine gültige Antwort geliefert hat.
        response: Provider-Antwort (leer bei Fehler).
        error_message: Benutzerfreundliche Fehlermeldung (leer bei Erfolg).
        error_id: Fehler-ID für Debugging (leer bei Erfolg).
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


class ChatService:
    """Hauptlogik für User-Anfragen.

    Koordiniert Provider-Aufruf, Memory-Loading, Conversation-History
    und Sprach-Detection. Ersetzt die alten Modul-Globals + Setter
    durch saubere Konstruktor-Injection.
    """

    def __init__(
        self,
        provider_router: "ProviderRouter",
        memory_service: "MemoryService | None" = None,
        model_service: "ModelService | None" = None,
        task_router: "TaskRouter | None" = None,
    ) -> None:
        self.provider_router = provider_router
        self.memory_service = memory_service
        self.model_service = model_service
        self.task_router = task_router

    def _build_all_slot_infos(self, user_id: int) -> list[SlotInfo]:
        """Baut die Slot-Belegungsliste für alle 6 Task-Slots.

        Prioritaet pro Slot:
          1. Slot-spezifischer Override
          2. Globaler Override
          3. Slot-Default (aus TaskRouter)
          4. System-Default

        Args:
            user_id: Telegram-User-ID.

        Returns:
            Liste von SlotInfo für alle 6 Slots.
        """
        from application.model_registry import ModelRegistry
        from application.model_service import DEFAULT_MODEL, resolve_alias
        from domain.task_slot import TaskSlot

        registry = ModelRegistry()
        result: list[SlotInfo] = []

        overrides: dict[str, str] = {}
        if self.model_service is not None:
            overrides = self.model_service.get_all_slot_overrides(user_id)

        global_override = overrides.get("global")

        slot_defaults: dict[str, str] = {}
        if self.task_router is not None and hasattr(
            self.task_router, "get_slot_defaults"
        ):
            for slot_enum, alias in self.task_router.get_slot_defaults().items():
                resolved = resolve_alias(alias)
                if resolved:
                    slot_defaults[slot_enum.value] = resolved

        for slot in TaskSlot:
            slot_override = overrides.get(slot.value)
            if slot_override:
                model_id = slot_override
                source = "user-override"
            elif global_override:
                model_id = global_override
                source = "global"
            else:
                model_id = slot_defaults.get(slot.value) or DEFAULT_MODEL
                source = "default"

            meta = registry.get(model_id)
            display_name = meta.display_name if meta else model_id
            result.append(
                SlotInfo(
                    slot_name=slot.value,
                    model_display_name=display_name,
                    source=source,
                )
            )

        return result

    def _build_self_awareness(
        self,
        user_model: str | None,
        task_slot_name: str | None,
        user_id: int | None = None,
    ) -> str:
        """Baut den Self-Awareness-Block für den System-Prompt.

        Resolved Modell-Metadaten aus der ModelRegistry und baut den Block.
        Wenn kein Modell resolved wurde, wird der System-Default verwendet.
        Wenn user_id gegeben, werden alle 6 Slot-Belegungen inkludiert.

        Args:
            user_model: Resolved Modell-ID oder None.
            task_slot_name: Name des Task-Slots oder None.
            user_id: Telegram-User-ID für Slot-Belegungsliste (optional).

        Returns:
            Self-Awareness-Block als String, oder leerer String bei Fehler.
        """
        from application.model_registry import ModelRegistry
        from application.model_service import DEFAULT_MODEL

        model_id = user_model or DEFAULT_MODEL
        slot = task_slot_name or "chat"

        try:
            registry = ModelRegistry()
            metadata = registry.get(model_id)

            # Alle 6 Slot-Belegungen sammeln (wenn user_id vorhanden)
            all_slots: list[SlotInfo] | None = None
            if user_id is not None:
                try:
                    all_slots = self._build_all_slot_infos(user_id)
                except Exception:
                    log.debug(
                        "Slot-Belegungsliste konnte nicht gebaut werden",
                        exc_info=True,
                    )

            if metadata is not None:
                return build_self_awareness_block(
                    model_display_name=metadata.display_name,
                    model_id=metadata.id,
                    task_slot=slot,
                    provider=metadata.provider,
                    all_slots=all_slots,
                )
            # Fallback: ID direkt verwenden wenn nicht in Registry
            return build_self_awareness_block(
                model_display_name=model_id,
                model_id=model_id,
                task_slot=slot,
                provider="unknown",
                all_slots=all_slots,
            )
        except Exception:
            log.debug("Self-Awareness-Block konnte nicht gebaut werden", exc_info=True)
            return ""

    def _get_memory_budget(self, provider_name: str | None = None) -> int:
        """Liest das Memory-Budget aus ProviderCapabilities.

        Falls der Provider eine eigene max_memory_chars-Capability definiert,
        wird diese verwendet. Sonst Fallback auf MAX_MEMORY_TOTAL_CHARS.

        Args:
            provider_name: Name des Providers (None = Default aus Router).

        Returns:
            Max. Zeichen für Memory-Block im System-Prompt.
        """
        if self.provider_router is None:
            return MAX_MEMORY_TOTAL_CHARS
        try:
            name = provider_name or self.provider_router.default
            providers_dict = self.provider_router.providers
            if not isinstance(providers_dict, dict):
                return MAX_MEMORY_TOTAL_CHARS
            provider = providers_dict.get(name)
            if provider:
                caps = provider.get_capabilities()
                value = getattr(caps, "max_memory_chars", MAX_MEMORY_TOTAL_CHARS)
                if isinstance(value, int):
                    return value
        except (AttributeError, TypeError):
            pass
        return MAX_MEMORY_TOTAL_CHARS

    def _build_memory_context(
        self, user_id: int, query: str, provider_name: str | None = None
    ) -> tuple[str, int]:
        """Lädt relevante Memory-Einträge für den User basierend auf Query.

        Such-Strategie: Keyword-Extraktion -> Substring-Match über alle Layer.
        Priorisiert längste Keywords als Hauptsuche.

        Args:
            user_id: Telegram-User-ID.
            query: Aktuelle User-Nachricht.
            provider_name: Optionaler Provider-Name für Memory-Budget-Lookup.

        Returns:
            Tuple von (Memory-Context-String, Anzahl geladener Entries).
            Leerer String + 0 wenn keine Treffer oder kein MemoryService.
        """
        if self.memory_service is None:
            return "", 0

        keywords = _extract_keywords(query)
        if not keywords:
            return "", 0

        # Primärer Suchbegriff: längstes Keyword (spezifischste)
        primary_query = keywords[0]

        episodic = self.memory_service.recall(
            user_id, primary_query, layer="episodic", limit=3
        )
        semantic = self.memory_service.recall(
            user_id, primary_query, layer="semantic", limit=3
        )
        procedural = self.memory_service.recall(
            user_id, primary_query, layer="procedural", limit=2
        )

        # Wenn primärer keine Treffer: sekundären probieren (falls vorhanden)
        if not (episodic or semantic or procedural) and len(keywords) > 1:
            secondary_query = keywords[1]
            episodic = self.memory_service.recall(
                user_id, secondary_query, layer="episodic", limit=3
            )
            semantic = self.memory_service.recall(
                user_id, secondary_query, layer="semantic", limit=3
            )
            procedural = self.memory_service.recall(
                user_id, secondary_query, layer="procedural", limit=2
            )

        if not (episodic or semantic or procedural):
            return "", 0

        total_entries = len(episodic) + len(semantic) + len(procedural)

        sections: list[str] = ["[GESPEICHERTE NOTIZEN]", ""]
        sections.append(
            "Diese Einträge wurden vom User gespeichert und sind möglicherweise "
            "relevant für die aktuelle Frage. Nutze sie wenn passend, ignoriere wenn nicht relevant."
        )
        sections.append("")

        if episodic:
            sections.append("Episodic (was passiert ist):")
            for entry in episodic:
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                sections.append(f"  • [{entry['id']}] {content}")
            sections.append("")

        if semantic:
            sections.append("Semantic (Fakten):")
            for entry in semantic:
                category = entry.get("category", "")
                cat_part = f" (kategorie: {category})" if category else ""
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                sections.append(f"  • [{entry['id']}]{cat_part} {content}")
            sections.append("")

        if procedural:
            sections.append("Procedural (Skills):")
            for entry in procedural:
                skill = entry.get("skill_name", "")
                skill_part = f" [skill: {skill}]" if skill else ""
                content = _truncate(entry["content"], MAX_MEMORY_CHARS_PER_ENTRY)
                sections.append(f"  • [{entry['id']}]{skill_part} {content}")
            sections.append("")

        memory_block = "\n".join(sections)
        budget = self._get_memory_budget(provider_name)
        if len(memory_block) > budget:
            memory_block = memory_block[: budget - 200] + "\n[Block gekürzt]"
        return memory_block, total_entries

    async def process_user_message(
        self,
        text: str,
        user_id: int | None,
        chat_id: int | None,
        username: str | None,
        system_prompt: str,
        language_override: Optional[str] = None,
        provider_name: Optional[str] = None,
        reply_to_text: Optional[str] = None,
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
            reply_to_text: Text der Nachricht auf die der User geantwortet hat (Telegram Reply-To).

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
            # Load conversation history
            history = await get_history(uid, cid)

            # Auto-Memory-Loading: relevante Einträge für aktuelle Frage laden
            memory_context, memory_entries_loaded = self._build_memory_context(
                uid, text
            )

            # Build context-enriched prompt (history + current message)
            # If user replied to a specific bot message, prepend that context
            if reply_to_text:
                enriched_text = (
                    "[USER REPLIED TO PREVIOUS BOT MESSAGE]\n"
                    f'"{reply_to_text}"\n\n'
                    "[USER'S CURRENT MESSAGE]\n"
                    f"{text}"
                )
            else:
                enriched_text = text
            context_prompt = build_context_block(history, enriched_text)

            # Language: Smart-Detection bei jeder Nachricht.
            # Sticky wird überschrieben wenn Detection klar genug ist (Variante 1).
            if language_override:
                lang = language_override
            else:
                sticky_lang = await get_language(uid, cid)
                detected_lang, confidence = detect_language_with_confidence(text)

                if not sticky_lang:
                    # First turn: detect and set sticky
                    lang = detected_lang if confidence > 0 else "de"
                    await set_language(uid, cid, lang)
                elif confidence > 0.7 and detected_lang != sticky_lang:
                    # User hat implizit die Sprache gewechselt
                    lang = detected_lang
                    await set_language(uid, cid, lang)
                    log.info(
                        "Smart-Language-Switch: %s -> %s (confidence=%.2f)",
                        sticky_lang,
                        lang,
                        confidence,
                    )
                else:
                    lang = sticky_lang

            if lang != "de":
                log.info("Sprache für Chat: '%s' (sticky)", lang)

            # Effektiven Prompt mit Language-Override bauen
            effective_prompt = build_effective_prompt(system_prompt, lang)

            # Memory-Context in System-Prompt einfügen (vor dem LLM-Call)
            if memory_context:
                effective_prompt = f"{effective_prompt}\n\n{memory_context}"

            # Phase 2a: TaskRouter-Klassifikation + Modell-Resolution
            task_slot_name: str | None = None
            task_score: int = 0
            task_matched_patterns: tuple[str, ...] = ()
            task_matched_keywords: tuple[str, ...] = ()
            user_model: str | None = None

            if self.task_router is not None:
                classification = self.task_router.classify(text)
                task_slot_name = classification.slot.value
                task_score = classification.score
                task_matched_patterns = classification.matched_patterns
                task_matched_keywords = classification.matched_keywords

                # Modell via TaskRouter resolven (Slot-Override > Global > Default)
                user_model = self.task_router.resolve_model(uid, classification.slot)
            elif self.model_service is not None:
                # Fallback: Phase 1 Verhalten (nur globaler Override)
                user_model = self.model_service.get_user_model(uid)

            # Self-Awareness-Block: Modell-Info in System-Prompt injizieren
            self_awareness = self._build_self_awareness(user_model, task_slot_name, uid)
            if self_awareness:
                effective_prompt = f"{effective_prompt}\n\n{self_awareness}"

            # Provider-Router aufrufen (ersetzt direkten claude_cli Aufruf)
            result = await self.provider_router.route(
                prompt=context_prompt,
                system_prompt=effective_prompt,
                provider_name=provider_name,
                user_id=uid,
                chat_id=cid,
                model=user_model,
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

            # Task-Classification ins Audit (Phase 2a Konfidenz-Logging)
            if task_slot_name is not None:
                audit["task_slot"] = task_slot_name
                audit["task_score"] = task_score
                audit["task_matched_patterns"] = list(task_matched_patterns)
                audit["task_matched_keywords"] = list(task_matched_keywords)
                if user_model:
                    audit["resolved_model"] = user_model

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

            # C-3: System-Prompt-Leakage-Guard
            leak_replacement = check_for_system_prompt_leakage(
                response, effective_prompt
            )
            if leak_replacement is not None:
                log.warning(
                    "Leakage-Filter hat Response ersetzt (user_id=%s, chat_id=%s)",
                    user_id,
                    chat_id,
                )
                audit["leakage_attempt"] = True
                response = leak_replacement

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
            log.error("CLI nicht gefunden. PATH prüfen.")
            return ChatResult(
                success=False,
                error_message=(
                    "Fehler: LLM CLI ist nicht installiert oder nicht im PATH.\n"
                    "Prüfe ob das entsprechende CLI im Terminal funktioniert."
                ),
            )

        except asyncio.TimeoutError:
            audit["error"] = "timeout"
            log.warning("Provider Timeout")
            return ChatResult(
                success=False,
                error_message=(
                    "Provider hat zu lange gebraucht. Versuche eine kürzere Frage."
                ),
            )

        except ProviderError as e:
            # Spezifische Provider-Fehler (Unavailable, NotImplemented, Timeout)
            error_id = uuid.uuid4().hex[:8]
            audit["error"] = f"provider_error: {e}"
            audit["error_id"] = error_id
            log.error(
                "Provider-Fehler (error_id=%s, provider=%s, retryable=%s): %s",
                error_id,
                e.provider_name,
                e.retryable,
                e,
            )
            hint = " Versuch es gleich noch mal." if e.retryable else ""
            return ChatResult(
                success=False,
                error_message=(
                    f"Der Sprachmodell-Anbieter meldet ein Problem "
                    f"(ref: {error_id}).{hint}"
                ),
                error_id=error_id,
            )

        except ValueError as e:
            # Provider nicht registriert o.ä.
            error_id = uuid.uuid4().hex[:8]
            audit["error"] = f"value_error: {e}"
            audit["error_id"] = error_id
            log.error("ValueError (error_id=%s): %s", error_id, e)
            return ChatResult(
                success=False,
                error_message=f"Anfrage konnte nicht verarbeitet werden (ref: {error_id}).",
                error_id=error_id,
            )

        except RuntimeError as e:
            # Fallback für unerwartete Runtime-Fehler
            error_id = uuid.uuid4().hex[:8]
            audit["error"] = f"runtime_error: {e}"
            audit["error_id"] = error_id
            log.error("RuntimeError (error_id=%s): %s", error_id, e)
            return ChatResult(
                success=False,
                error_message=f"Interner Fehler (ref: {error_id}).",
                error_id=error_id,
            )

        except Exception as e:
            error_id = uuid.uuid4().hex[:8]
            log.exception(
                "Unbekannter Fehler bei der Verarbeitung (error_id=%s)", error_id
            )
            audit["error"] = str(e)
            audit["error_id"] = error_id
            return ChatResult(
                success=False,
                error_message=f"Ein interner Fehler ist aufgetreten. Fehler-ID: {error_id}",
                error_id=error_id,
            )

        finally:
            write_audit_log(audit)

    async def process_user_message_streaming(
        self,
        text: str,
        user_id: int | None,
        chat_id: int | None,
        username: str | None,
        system_prompt: str,
        persistent_provider: "ClaudePersistentProvider",
        language_override: Optional[str] = None,
        reply_to_text: Optional[str] = None,
        status_session: Optional[Any] = None,
    ) -> tuple[AsyncIterator[StreamEvent], int]:
        """Streaming-Variante von process_user_message.

        Bereitet Prompt identisch vor (Memory, History, Language),
        nutzt aber den ClaudePersistentProvider für Token-Streaming.
        History-Speicherung und Audit passieren NACH dem Stream.

        Args:
            text: User-Nachricht.
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID (auch Process-Routing-Key).
            username: Telegram Username.
            system_prompt: Base System-Prompt.
            persistent_provider: Der Streaming-fähige Provider.
            language_override: Optionale Sprach-Override.
            reply_to_text: Reply-To-Kontext.
            status_session: Optionale StatusSession für Status-Updates.

        Returns:
            Tuple von (StreamEvent-AsyncIterator, memory_entries_loaded).
            memory_entries_loaded: Anzahl Memory-Einträge die in den Prompt
            eingefügt wurden (für Audit-Log).

        Note:
            Der Aufrufer muss nach dem Stream selbst:
            1. save_streaming_result() aufrufen für History + Audit
        """
        uid = user_id or 0
        cid = chat_id or 0

        # Status: Memory-Loading
        if status_session is not None:
            await status_session.update("memory_loading")

        # Prompt vorbereiten (identisch zu process_user_message)
        history = await get_history(uid, cid)
        memory_context, memory_entries_loaded = self._build_memory_context(uid, text)

        # Status: Memory geladen (mit Anzahl)
        if status_session is not None and memory_entries_loaded > 0:
            await status_session.update("memory_loaded", n=memory_entries_loaded)

        if reply_to_text:
            enriched_text = (
                "[USER REPLIED TO PREVIOUS BOT MESSAGE]\n"
                f'"{reply_to_text}"\n\n'
                "[USER'S CURRENT MESSAGE]\n"
                f"{text}"
            )
        else:
            enriched_text = text
        context_prompt = build_context_block(history, enriched_text)

        # Language: Smart-Detection bei jeder Nachricht (Variante 1).
        if language_override:
            lang = language_override
        else:
            sticky_lang = await get_language(uid, cid)
            detected_lang, confidence = detect_language_with_confidence(text)

            if not sticky_lang:
                lang = detected_lang if confidence > 0 else "de"
                await set_language(uid, cid, lang)
            elif confidence > 0.7 and detected_lang != sticky_lang:
                # User hat implizit die Sprache gewechselt
                lang = detected_lang
                await set_language(uid, cid, lang)
                log.info(
                    "Smart-Language-Switch (streaming): %s -> %s (confidence=%.2f)",
                    sticky_lang,
                    lang,
                    confidence,
                )
            else:
                lang = sticky_lang

        # Sprache der StatusSession aktualisieren (Bug-Fix: Sticky-Language
        # wird erst hier bestimmt, aber StatusSession wurde vorher erstellt)
        if status_session is not None:
            status_session.set_language(lang)

        effective_prompt = build_effective_prompt(system_prompt, lang)
        if memory_context:
            effective_prompt = f"{effective_prompt}\n\n{memory_context}"

        # Phase 2a: TaskRouter-Klassifikation + Modell-Resolution
        user_model: str | None = None
        task_slot_name: str | None = None
        if self.task_router is not None:
            classification = self.task_router.classify(text)
            task_slot_name = classification.slot.value
            user_model = self.task_router.resolve_model(uid, classification.slot)
        elif self.model_service is not None:
            # Fallback: Phase 1 Verhalten (nur globaler Override)
            user_model = self.model_service.get_user_model(uid)

        # Self-Awareness-Block: Modell-Info in System-Prompt injizieren
        self_awareness = self._build_self_awareness(user_model, task_slot_name, uid)
        if self_awareness:
            effective_prompt = f"{effective_prompt}\n\n{self_awareness}"

        # Status: Denke nach (vor Provider-Call)
        if status_session is not None:
            await status_session.update("thinking")

        # Streaming via persistent Provider
        async def _stream() -> AsyncIterator[StreamEvent]:
            first_token = True
            async for event in persistent_provider.query_streaming(
                prompt=context_prompt,
                system_prompt=effective_prompt,
                user_id=uid,
                chat_id=cid,
                model=user_model,
            ):
                # Bei erstem Token: Status-Updates stoppen
                if first_token and event.event_type == "content_delta":
                    first_token = False
                    if status_session is not None:
                        status_session.mark_stream_started()
                yield event

        return _stream(), memory_entries_loaded

    async def save_streaming_result(
        self,
        user_id: int,
        chat_id: int,
        user_text: str,
        response_text: str,
        duration_seconds: float,
        username: str | None = None,
        was_cold: bool = False,
        streaming_chunks: int = 0,
        subprocess_pid: int = 0,
        memory_entries_loaded: int = 0,
        system_prompt: str = "",
    ) -> str:
        """Speichert das Ergebnis einer Streaming-Session in History + Audit.

        Wird vom Presentation-Layer NACH Abschluss des Streams aufgerufen.
        Prüft Response auf System-Prompt-Leakage (C-3) und gibt die
        ggf. bereinigte Response zurück.

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            user_text: Original User-Nachricht.
            response_text: Vollständige Provider-Antwort.
            duration_seconds: Gesamtdauer der Anfrage.
            username: Telegram Username.
            was_cold: True wenn ein neuer Subprocess gestartet wurde.
            streaming_chunks: Anzahl empfangener Content-Delta-Events.
            subprocess_pid: PID des genutzten Subprocess.
            memory_entries_loaded: Anzahl geladener Memory-Einträge.
            system_prompt: Aktiver System-Prompt für Leakage-Check (C-3).

        Returns:
            Die (ggf. bereinigte) Response-Text. Caller muss prüfen ob
            sich der Text geändert hat für ein finales Telegram-Edit.
        """
        leakage_detected = False

        # C-3: Leakage-Check auf die finale Response
        if system_prompt:
            leak_replacement = check_for_system_prompt_leakage(
                response_text, system_prompt
            )
            if leak_replacement is not None:
                log.warning(
                    "Leakage-Filter (Streaming): Response ersetzt "
                    "(user_id=%s, chat_id=%s)",
                    user_id,
                    chat_id,
                )
                response_text = leak_replacement
                leakage_detected = True

        # History speichern
        user_turn = ConversationTurn(role="user", content=user_text)
        assistant_turn = ConversationTurn(role="assistant", content=response_text)
        await save_turn(user_id, chat_id, user_turn)
        await save_turn(user_id, chat_id, assistant_turn)

        # Audit-Log
        audit: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "stream_completed",
            "user_id": user_id,
            "chat_id": chat_id,
            "username": username,
            "prompt_length": len(user_text),
            "provider": "claude_persistent",
            "response_length": len(response_text),
            "duration_seconds": round(duration_seconds, 2),
            "was_warm": not was_cold,
            "was_cold": was_cold,
            "streaming_chunks": streaming_chunks,
            "subprocess_pid": subprocess_pid,
            "memory_entries_loaded": memory_entries_loaded,
        }
        if leakage_detected:
            audit["leakage_attempt"] = True
        write_audit_log(audit)

        return response_text

    async def reset(self, user_id: int, chat_id: int) -> None:
        """Use-Case-Wrapper: setzt Conversation und Sticky-Language zurück."""
        await _infra_reset_conversation(user_id, chat_id)

    async def get_chat_language(self, user_id: int, chat_id: int) -> str | None:
        """Use-Case-Wrapper: liest die Sticky-Language für einen Chat."""
        return await get_language(user_id, chat_id)

    async def set_chat_language(self, user_id: int, chat_id: int, lang: str) -> None:
        """Use-Case-Wrapper: setzt die Sticky-Language für einen Chat."""
        await set_language(user_id, chat_id, lang)

    async def save_static_response_to_history(
        self, user_id: int, chat_id: int, response_text: str
    ) -> None:
        """Speichert eine statische Bot-Antwort (z.B. /start, /help) in die History.

        Damit weiß der Bot beim nächsten Turn was er gerade gesagt hat.
        Nur assistant-Turn wird gespeichert (der User-Command selbst ist kein
        natürlicher Konversationsbeitrag).

        Args:
            user_id: Telegram User-ID.
            chat_id: Telegram Chat-ID.
            response_text: Der gesendete Bot-Text.
        """
        turn = ConversationTurn(role="assistant", content=response_text)
        await save_turn(user_id, chat_id, turn)
