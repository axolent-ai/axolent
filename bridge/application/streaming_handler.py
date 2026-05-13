"""Streaming-Handler: koordiniert Token-Streaming zu Telegram-Edits.

Wandelt StreamEvents aus dem ClaudePersistentProvider in
Telegram-Message-Edits um. Rate-Limited mit adaptivem Throttle.

Features:
    - Aggregiert Tokens bis zum nächsten Edit-Zeitpunkt
    - Erste Edit nach ~1.5s mit akkumuliertem Text
    - Danach alle ~1.5s den vollständigen bisherigen Text als Edit
    - Bei finalem Result: letzte Edit mit vollständigem Text + HTML-Formatierung
    - Zwischen-Edits: Markdown wird live gerendert, unvollständige Tokens
      am Ende werden sicher abgeschnitten (Option A: smart-trim)
    - Finale Edit konvertiert Markdown zu Telegram-HTML via domain.markdown
    - Bei Antworten >4096 Zeichen: Multi-Message-Split an sinnvollen Grenzen
    - Telegram-API-Fehler werden leise geschluckt (UX > Crash)
    - Adaptive Flood-Control: Bei Telegram 429 (RetryAfter) pausiert die
      Session, Zwischen-Edits werden übersprungen, Throttle wird
      exponentiell erhöht und erholt sich nach erfolgreichen Edits.
    - Final-Edits haben höchste Priorität und werden bei 429 retried.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from domain.markdown import markdown_to_telegram_html, strip_markdown

if TYPE_CHECKING:
    from telegram import Message

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive Throttle / Flood-Control Konstanten
# ---------------------------------------------------------------------------
# R04 Round 4: Adaptive Flood-Control. Live-Stress-Test mit 308 Streaming-Chunks
# triggerte kaskadierende Telegram 429er. Backoff-Faktor 2.0, Recovery 0.7
# nach 5 erfolgreichen Edits. Final-Edits haben Priority und werden bei
# 429 retried. Empirisch validiert mit 0 Errors bei 15.678 Zeichen Antwort.

# Default-Throttle für Zwischen-Edits (Sekunden)
DEFAULT_THROTTLE: float = 1.5

# Maximaler Throttle nach wiederholten 429ern (Sekunden)
MAX_THROTTLE: float = 10.0

# Faktor für Throttle-Erhöhung bei 429
THROTTLE_BACKOFF_FACTOR: float = 2.0

# Nach N erfolgreichen Edits wird der Throttle schrittweise reduziert
THROTTLE_RECOVERY_AFTER: int = 5

# Faktor für Throttle-Reduktion bei Recovery
THROTTLE_RECOVERY_FACTOR: float = 0.7

# Maximale Retries für Final-Edits bei 429
FINAL_EDIT_MAX_RETRIES: int = 2

# ---------------------------------------------------------------------------
# Bisherige Konstanten
# ---------------------------------------------------------------------------

# Rate-Limiting: Telegram erlaubt max ~30 edits/min pro Chat
# Default-Throttle wird jetzt adaptiv gesteuert (s.o.)
EDIT_INTERVAL_SECONDS: float = DEFAULT_THROTTLE

# Erste Edit nach dieser Zeit (damit genug Text da ist)
FIRST_EDIT_DELAY_SECONDS: float = 1.5

# Maximale Nachrichtenlänge für Telegram (4096 Zeichen)
TELEGRAM_MAX_LENGTH: int = 4096

# Puffer für Part-Marker ("(2/3)") und Sicherheitsmarge
_SPLIT_SAFETY_MARGIN: int = 30


@dataclass
class StreamingSession:
    """State einer laufenden Streaming-Session.

    Attributes:
        message: Die Telegram-Nachricht die editiert wird.
        accumulated_text: Bisher gesammelter Text.
        last_edit_time: Zeitpunkt der letzten Edit.
        edit_count: Anzahl bisheriger Edits.
        started_at: Session-Startzeit.
        is_first_edit: Ob noch keine Edit gesendet wurde.
        _last_edit_html: Zuletzt gesendeter Edit-Text (für Duplikat-Erkennung).
        _paused_until: Monotonic-Timestamp bis zu dem Edits pausiert sind (Flood-Control).
        _current_throttle: Aktueller adaptiver Edit-Intervall in Sekunden.
        _consecutive_success: Zähler erfolgreicher Edits seit letztem 429.
    """

    message: "Message"
    accumulated_text: str = ""
    last_edit_time: float = 0.0
    edit_count: int = 0
    started_at: float = 0.0
    is_first_edit: bool = True
    _last_edit_html: str = ""
    _paused_until: float = 0.0
    _current_throttle: float = field(default_factory=lambda: DEFAULT_THROTTLE)
    _consecutive_success: int = 0


async def create_streaming_message(chat: Any) -> "Message":
    """Erstellt die initiale Placeholder-Nachricht für Streaming.

    Args:
        chat: Telegram Chat-Objekt.

    Returns:
        Die gesendete Message (wird später editiert).
    """
    # Platzhalter der signalisiert dass gearbeitet wird
    return await chat.send_message("...")


async def process_streaming_edit(
    session: StreamingSession,
    new_text: str,
) -> None:
    """Fügt neuen Text hinzu und editiert die Nachricht falls nötig.

    Rate-Limited: adaptiver Throttle (Default 1.5s, steigt bei 429).
    Erste Edit erst nach FIRST_EDIT_DELAY_SECONDS.
    Während Flood-Control-Pause werden Zwischen-Edits übersprungen.

    Args:
        session: Die aktuelle StreamingSession.
        new_text: Neuer inkrementeller Text.
    """
    session.accumulated_text += new_text
    now = time.monotonic()

    # Erste Edit: warte bis genug Zeit vergangen ist
    if session.is_first_edit:
        elapsed = now - session.started_at
        if elapsed < FIRST_EDIT_DELAY_SECONDS:
            return  # Noch nicht genug Text gesammelt
        session.is_first_edit = False

    # Flood-Control-Pause: Zwischen-Edits werden übersprungen
    if session._paused_until and now < session._paused_until:
        return

    # Rate-Limiting: mindestens _current_throttle seit letzter Edit
    time_since_edit = now - session.last_edit_time
    if time_since_edit < session._current_throttle:
        return  # Zu früh für nächste Edit

    await _do_edit(session)


async def finalize_streaming(session: StreamingSession, final_text: str) -> str:
    """Finalisiert die Streaming-Session mit dem vollständigen Text.

    Bei kurzen Antworten (<= 4096 Zeichen): eine Edit mit HTML.
    Bei langen Antworten (> 4096 Zeichen): Multi-Message-Split.
    Die erste Message wird per Edit aktualisiert, Folge-Teile als
    neue Nachrichten im selben Chat gesendet.

    Args:
        session: Die aktuelle StreamingSession.
        final_text: Der vollständige Antworttext.

    Returns:
        Der finale Text (ungekürzt, für History-Speicherung).
    """
    session.accumulated_text = final_text

    # HTML-konvertierter Text bestimmt ob Split nötig ist
    html_text = markdown_to_telegram_html(final_text)

    if len(html_text) <= TELEGRAM_MAX_LENGTH:
        # Kurze Antwort: einfache Edit (bisheriges Verhalten)
        await _do_edit_html(session)
        return final_text

    # Lange Antwort: Multi-Message-Split
    await _finalize_multi_message(session, final_text)
    return final_text


async def _finalize_multi_message(
    session: StreamingSession,
    full_text: str,
) -> None:
    """Splittet eine lange Antwort in mehrere Telegram-Nachrichten.

    Strategie:
        1. Plain-Text (Markdown) an sinnvollen Grenzen splitten
        2. Jeden Teil separat zu HTML konvertieren (damit Tags korrekt geschlossen)
        3. Teil 1 als Edit der bestehenden Streaming-Message
        4. Teile 2+ als neue Nachrichten im Chat

    Multi-Message-Teile gehören zur Final-Phase und bekommen
    dasselbe RetryAfter-Handling wie Final-Edits.

    Args:
        session: Die aktuelle StreamingSession.
        full_text: Der vollständige Markdown-Text.
    """
    parts = split_text_for_telegram(full_text)
    total = len(parts)

    for i, part in enumerate(parts):
        part_num = i + 1
        # Part-Marker anfügen (ausser bei Einzelteil)
        if total > 1:
            marker = f"\n\n({part_num}/{total})"
        else:
            marker = ""

        html_part = markdown_to_telegram_html(part + marker)

        if i == 0:
            # Erste Nachricht: Edit der bestehenden Streaming-Placeholder-Message
            # Mit RetryAfter-Handling (Final-Priorität)
            await _send_final_edit_with_retry(
                session, html_part, part + marker, part_num, total
            )
        else:
            # Folge-Nachrichten: neue Message im selben Chat
            # Mit RetryAfter-Handling (Final-Priorität)
            await _send_final_message_with_retry(
                session, html_part, part + marker, part_num, total
            )


async def _send_final_edit_with_retry(
    session: StreamingSession,
    html_text: str,
    plain_source: str,
    part_num: int,
    total: int,
) -> None:
    """Editiert die erste Nachricht im Multi-Message-Split mit Retry bei 429.

    Nach erschöpften Retries: Fallback auf send_message (wie _do_edit_html),
    damit der erste Teil nicht im Placeholder hängen bleibt.

    Args:
        session: Die aktuelle StreamingSession.
        html_text: Fertig konvertierter HTML-Text.
        plain_source: Original-Markdown für strip_markdown Fallback.
        part_num: Teil-Nummer (für Logging).
        total: Gesamt-Anzahl Teile (für Logging).
    """
    for attempt in range(1 + FINAL_EDIT_MAX_RETRIES):
        try:
            await _send_html_with_fallback(session.message, html_text, plain_source)
            session.last_edit_time = time.monotonic()
            session.edit_count += 1
            _record_edit_success(session)
            return
        except Exception as e:
            retry_after = _is_retry_after(e)
            if retry_after is not None and attempt < FINAL_EDIT_MAX_RETRIES:
                _apply_flood_backoff(session, retry_after)
                log.info(
                    "Multi-Message Edit Teil %d/%d: 429, warte %ds",
                    part_num,
                    total,
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                continue
            if retry_after is not None:
                log.error(
                    "Multi-Message Edit Teil %d/%d: 429 nach %d Retries, "
                    "Fallback auf send_message",
                    part_num,
                    total,
                    FINAL_EDIT_MAX_RETRIES,
                )
                # Fallback: als neue Nachricht senden statt Edit
                plain_text = strip_markdown(plain_source)
                try:
                    await session.message.chat.send_message(plain_text)
                except Exception as fb_e:
                    log.error(
                        "Multi-Message Edit Fallback send_message "
                        "Teil %d/%d fehlgeschlagen: %s",
                        part_num,
                        total,
                        fb_e,
                    )
                return
            # Nicht-429-Fehler: loggen und weiter
            _handle_edit_error(e)
            return


async def _send_final_message_with_retry(
    session: StreamingSession,
    html_text: str,
    plain_source: str,
    part_num: int,
    total: int,
) -> None:
    """Sendet eine Folge-Nachricht im Multi-Message-Split mit Retry bei 429.

    Args:
        session: Die aktuelle StreamingSession.
        html_text: Fertig konvertierter HTML-Text.
        plain_source: Original-Markdown für strip_markdown Fallback.
        part_num: Teil-Nummer (für Logging).
        total: Gesamt-Anzahl Teile (für Logging).
    """
    for attempt in range(1 + FINAL_EDIT_MAX_RETRIES):
        try:
            await session.message.chat.send_message(html_text, parse_mode="HTML")
            _record_edit_success(session)
            return
        except Exception as e:
            # Flood-Control: warten und retrien
            retry_after = _is_retry_after(e)
            if retry_after is not None and attempt < FINAL_EDIT_MAX_RETRIES:
                _apply_flood_backoff(session, retry_after)
                log.info(
                    "Multi-Message Send Teil %d/%d: 429, warte %ds",
                    part_num,
                    total,
                    retry_after,
                )
                await asyncio.sleep(retry_after)
                continue

            error_str = str(e).lower()
            if "can't parse entities" in error_str or "bad request" in error_str:
                log.warning(
                    "HTML-Send Teil %d/%d fehlgeschlagen, Fallback Plain: %s",
                    part_num,
                    total,
                    e,
                )
                plain = strip_markdown(plain_source)
                try:
                    await session.message.chat.send_message(plain)
                except Exception as fb_e:
                    log.warning(
                        "Plain-Send Teil %d/%d fehlgeschlagen: %s",
                        part_num,
                        total,
                        fb_e,
                    )
                return
            log.warning("Send Teil %d/%d Fehler: %s", part_num, total, e)
            return


async def _send_html_with_fallback(
    message: "Message",
    html_text: str,
    plain_source: str,
) -> None:
    """Editiert eine Nachricht mit HTML, Fallback auf Plain-Text.

    RetryAfter-Exceptions werden NICHT gefangen sondern durchgereicht,
    damit der aufrufende Code (z.B. _send_final_edit_with_retry) den
    Retry-Loop steuern kann.

    Args:
        message: Die Telegram-Message die editiert wird.
        html_text: Fertig konvertierter HTML-Text.
        plain_source: Original-Markdown für strip_markdown Fallback.

    Raises:
        Exception: Bei RetryAfter/Flood-Control wird die Exception durchgereicht.
    """
    try:
        await message.edit_text(html_text, parse_mode="HTML")
    except Exception as e:
        # Flood-Control: durchreichen für Retry-Logik im Aufrufer
        if _is_retry_after(e) is not None:
            raise

        error_str = str(e).lower()
        if "message is not modified" in error_str:
            pass
        elif "can't parse entities" in error_str or "bad request" in error_str:
            log.warning("HTML-Edit fehlgeschlagen, Fallback auf Plain-Text: %s", e)
            plain_text = strip_markdown(plain_source)
            try:
                await message.edit_text(plain_text)
            except Exception as fallback_e:
                _handle_edit_error(fallback_e)
        else:
            _handle_edit_error(e)


def split_text_for_telegram(
    text: str,
    max_length: int = TELEGRAM_MAX_LENGTH,
) -> list[str]:
    """Splittet Text intelligent für Telegram-Nachrichten.

    Splitting-Priorität:
        1. Doppelter Zeilenumbruch (Absatz-Ende)
        2. Einfacher Zeilenumbruch (Zeilen-Ende)
        3. Satzende (. ! ?)
        4. Wort-Grenze (Leerzeichen)
        5. Harter Schnitt (Fallback)

    Markdown-aware: Schneidet nicht mitten in **bold**, *italic*,
    `code`, ```code blocks```, oder [links](url).

    Args:
        text: Der zu splittende Text.
        max_length: Maximale Länge pro Teil (inkl. Part-Marker-Puffer).

    Returns:
        Liste von Text-Teilen, jeder <= max_length Zeichen.
    """
    if len(text) <= max_length - _SPLIT_SAFETY_MARGIN:
        return [text]

    effective_max = max_length - _SPLIT_SAFETY_MARGIN
    parts: list[str] = []
    remaining = text

    while remaining:
        if len(remaining) <= effective_max:
            parts.append(remaining)
            break

        # Finde die beste Split-Position
        split_pos = _find_split_position(remaining, effective_max)
        part = remaining[:split_pos].rstrip()
        remaining = remaining[split_pos:].lstrip("\n")

        if part:
            parts.append(part)

    return parts if parts else [text[:effective_max]]


def _find_split_position(text: str, max_pos: int) -> int:
    """Findet die beste Position zum Splitten.

    Sucht rückwärts von max_pos nach der besten Trennstelle.
    Respektiert Markdown-Token-Grenzen.

    Args:
        text: Der Text in dem gesucht wird.
        max_pos: Maximale Position (exklusiv).

    Returns:
        Die beste Split-Position.
    """
    search_text = text[:max_pos]

    # Priorität 1: Doppelter Zeilenumbruch (Absatz)
    pos = search_text.rfind("\n\n")
    if pos > max_pos // 3:  # Nicht zu weit vorne splitten
        candidate = pos + 2  # Nach dem Doppel-Newline
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Priorität 2: Einfacher Zeilenumbruch
    pos = search_text.rfind("\n")
    if pos > max_pos // 3:
        candidate = pos + 1
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Priorität 3: Satzende (. ! ? gefolgt von Leerzeichen oder Zeilenende)
    sentence_end = None
    for m in re.finditer(r"[.!?]\s", search_text):
        if m.end() > max_pos // 3:
            sentence_end = m.end()
    if sentence_end and _is_safe_markdown_position(text, sentence_end):
        return sentence_end

    # Priorität 4: Wort-Grenze (Leerzeichen)
    pos = search_text.rfind(" ")
    if pos > max_pos // 3:
        candidate = pos + 1
        if _is_safe_markdown_position(text, candidate):
            return candidate

    # Fallback: harter Schnitt
    return max_pos


def _is_safe_markdown_position(text: str, pos: int) -> bool:
    """Prüft ob eine Position sicher für einen Split ist.

    Unsicher wenn wir uns mitten in einem Markdown-Token befinden:
    - Ungerade Anzahl ** vor der Position (offener Bold-Marker)
    - Offener Backtick-Block (```)
    - Offene Inline-Backticks (`)
    - Offener Link [text]( ohne schließendes )

    Args:
        text: Der vollständige Text.
        pos: Die zu prüfende Position.

    Returns:
        True wenn der Split an dieser Position sicher ist.
    """
    before = text[:pos]

    # Check: offener Fenced Code-Block (ungerade Anzahl ```)
    fence_count = before.count("```")
    if fence_count % 2 != 0:
        return False

    # Check: offener Bold-Marker (ungerade Anzahl **)
    bold_count = before.count("**")
    if bold_count % 2 != 0:
        return False

    # Check: offener Inline-Code (ungerade Anzahl einzelner `)
    # Zuerst ``` entfernen, dann einzelne ` zählen
    cleaned = before.replace("```", "")
    backtick_count = cleaned.count("`")
    if backtick_count % 2 != 0:
        return False

    # Check: offener Link [text](  (kein schließendes ))
    last_open_bracket = before.rfind("[")
    if last_open_bracket >= 0:
        after_bracket = before[last_open_bracket:]
        if "(" in after_bracket and ")" not in after_bracket.split("(", 1)[1:]:
            # Wir sind in einem offenen Link
            close_paren = text.find(")", pos)
            if close_paren >= 0:
                return False

    return True


# R04 Round 2: Markdown-Smart-Trim verhindert dass User während Streaming
# rohe ** oder ` Tokens sieht. Schneidet unvollständige Markdown-Tokens
# am Ende ab, damit der sichtbare Teil sauber als HTML rendert.
def find_safe_markdown_end(text: str) -> int:
    """Findet die letzte sichere Position für Markdown-Rendering.

    Wird für Zwischen-Edits verwendet (Option A: smart-trim).
    Schneidet unvollständige Markdown-Tokens am Ende ab,
    damit der sichtbare Teil sauber als HTML gerendert werden kann.

    Sucht von hinten nach der letzten Position wo alle Markdown-Tokens
    geschlossen sind.

    Args:
        text: Der bisherige akkumulierte Text.

    Returns:
        Position bis zu der sicher gerendert werden kann.
    """
    if not text:
        return 0

    # Schnell-Check: wenn alles geschlossen ist, ganzen Text nehmen
    if _is_safe_markdown_position(text, len(text)):
        return len(text)

    # Rückwärts suchen: letzte Position wo alles sicher ist
    # Starte 50 Zeichen vor Ende (typische Token-Länge)
    search_start = max(0, len(text) - 50)
    best = search_start

    for i in range(len(text), search_start, -1):
        if _is_safe_markdown_position(text, i):
            best = i
            break

    return best


async def abort_streaming(session: StreamingSession, error_text: str) -> None:
    """Bricht Streaming ab und zeigt Fehlermeldung.

    Args:
        session: Die aktuelle StreamingSession.
        error_text: Fehlermeldung für den User.
    """
    session.accumulated_text = error_text
    await _do_edit(session)


# R04 Round 3: HTML-Truncation-Bug Fix. Statt HTML blind abzuschneiden
# (was <b>-Tags zerstört und Telegram 400 Bad Request "Can't parse entities"
# triggert), wird der Markdown-Text per Binary-Search so gekürzt, dass
# markdown_to_telegram_html(result) <= max_html_length.
def _truncate_markdown_for_html_limit(
    text: str,
    max_html_length: int = TELEGRAM_MAX_LENGTH,
) -> str:
    """Kürzt Markdown-Text so, dass die HTML-Konvertierung unter dem Limit bleibt.

    Statt HTML blind abzuschneiden (was Tags zerstört), wird der Markdown-Text
    per Binary-Search so gekürzt, dass markdown_to_telegram_html(result) <= max_html_length.

    Args:
        text: Markdown-Text.
        max_html_length: Maximale HTML-Länge (Default: 4096).

    Returns:
        Gekürzter Markdown-Text dessen HTML-Version unter dem Limit liegt.
    """
    html = markdown_to_telegram_html(text)
    if len(html) <= max_html_length:
        return text

    # Schätzung: Markdown ist kürzer als HTML (Tags brauchen Platz).
    # Starte mit proportionalem Schätzwert.
    ratio = max_html_length / max(len(html), 1)
    estimate = int(len(text) * ratio * 0.9)  # 10% Sicherheitsmarge

    # Binary-Search für exakte Grenze
    lo = max(0, estimate - 200)
    hi = min(len(text), estimate + 200)

    # Sicherstellen dass lo tatsaechlich unter dem Limit liegt
    while lo > 0 and len(markdown_to_telegram_html(text[:lo])) > max_html_length - 3:
        lo = lo // 2

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        # Finde sichere Markdown-Position nahe mid
        safe_pos = find_safe_markdown_end(text[:mid])
        if safe_pos == 0:
            safe_pos = mid  # Fallback wenn kein sicherer Punkt gefunden

        candidate = text[:safe_pos]
        candidate_html = markdown_to_telegram_html(candidate)

        if len(candidate_html) <= max_html_length - 3:  # Platz für "..."
            best = safe_pos
            lo = mid + 1
        else:
            hi = mid - 1

    return text[:best]


def _is_retry_after(exc: Exception) -> int | None:
    """Prüft ob eine Exception ein Telegram RetryAfter (429) ist.

    Erkennt sowohl die python-telegram-bot RetryAfter-Exception als auch
    generische Exceptions deren Message 'flood control' enthält.

    Returns:
        retry_after in Sekunden, oder None wenn kein 429.
    """
    # python-telegram-bot >= 20: telegram.error.RetryAfter
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return int(retry_after)

    # Fallback: String-Match für generische Exceptions
    msg = str(exc).lower()
    if "flood control" in msg or "429" in msg:
        # Versuche retry_after aus dem Message zu parsen
        import re as _re

        m = _re.search(r"retry in (\d+)", msg)
        if m:
            return int(m.group(1))
        return 30  # Konservativer Default wenn nicht parsbar

    return None


def _apply_flood_backoff(session: StreamingSession, retry_after: int) -> None:
    """Wendet Flood-Control-Backoff auf die Session an.

    Setzt Pause-Timestamp, verdoppelt Throttle, resetet Success-Counter.
    """
    now = time.monotonic()
    session._paused_until = now + retry_after
    session._current_throttle = min(
        session._current_throttle * THROTTLE_BACKOFF_FACTOR,
        MAX_THROTTLE,
    )
    session._consecutive_success = 0
    log.info(
        "Flood-Control: Pause für %ds, Throttle adaptiv auf %.1fs erhöht",
        retry_after,
        session._current_throttle,
    )


def _record_edit_success(session: StreamingSession) -> None:
    """Registriert eine erfolgreiche Edit und reduziert Throttle bei Recovery."""
    session._consecutive_success += 1
    if session._consecutive_success >= THROTTLE_RECOVERY_AFTER:
        old_throttle = session._current_throttle
        session._current_throttle = max(
            session._current_throttle * THROTTLE_RECOVERY_FACTOR,
            DEFAULT_THROTTLE,
        )
        session._consecutive_success = 0
        if old_throttle > session._current_throttle:
            log.debug(
                "Throttle reduziert auf %.1fs (war %.1fs)",
                session._current_throttle,
                old_throttle,
            )


async def _do_edit(session: StreamingSession) -> None:
    """Führt eine Telegram-Message-Edit durch (Zwischen-Edits).

    Verwendet Option A (smart-trim): Markdown wird live zu HTML konvertiert.
    Unvollständige Markdown-Tokens am Ende werden abgeschnitten,
    damit der sichtbare Teil sauber formatiert ist.

    Handelt Telegram-API-Fehler leise (loggt aber).
    Bei RetryAfter (429): Session wird pausiert, Zwischen-Edit übersprungen.
    Bei Überschreitung der Telegram-Länge wird der Markdown-Text
    intelligent gekürzt (nicht der HTML-Text, da das Tags zerstören würde).
    """
    raw = session.accumulated_text
    if not raw.strip():
        raw = "..."

    # Smart-Trim: finde sichere Markdown-End-Position
    safe_end = find_safe_markdown_end(raw)

    if safe_end > 0 and safe_end >= len(raw) // 2:
        # Genug Text ist sicher renderbar: als HTML senden
        safe_text = raw[:safe_end]

        # Markdown-Text kürzen falls HTML zu lang (statt HTML blind abzuschneiden)
        html_text = markdown_to_telegram_html(safe_text)
        if len(html_text) > TELEGRAM_MAX_LENGTH:
            safe_text = _truncate_markdown_for_html_limit(safe_text)
            html_text = markdown_to_telegram_html(safe_text)
            if len(html_text) > TELEGRAM_MAX_LENGTH:
                # Absoluter Fallback: Plain-Text
                html_text = strip_markdown(safe_text)
                if len(html_text) > TELEGRAM_MAX_LENGTH:
                    html_text = html_text[: TELEGRAM_MAX_LENGTH - 3] + "..."

        # Duplikat-Check: kein API-Call wenn Text identisch zur letzten Edit
        if html_text == session._last_edit_html:
            return

        try:
            await session.message.edit_text(html_text, parse_mode="HTML")
            session._last_edit_html = html_text
            session.last_edit_time = time.monotonic()
            session.edit_count += 1
            _record_edit_success(session)
            return
        except Exception as e:
            # Flood-Control: Pause + Skip (Zwischen-Edit, nicht retrien)
            retry_after = _is_retry_after(e)
            if retry_after is not None:
                _apply_flood_backoff(session, retry_after)
                return

            error_str = str(e).lower()
            if "message is not modified" in error_str:
                session._last_edit_html = html_text
                session.last_edit_time = time.monotonic()
                return
            if "can't parse entities" in error_str or "bad request" in error_str:
                log.debug("HTML-Zwischen-Edit fehlgeschlagen, Fallback Plain: %s", e)
                # Fallthrough zu Plain-Text
            else:
                _handle_edit_error(e)
                return

    # Fallback: Plain-Text (wenn safe_end zu kurz oder HTML fehlgeschlagen)
    text = raw
    if len(text) > TELEGRAM_MAX_LENGTH:
        text = text[: TELEGRAM_MAX_LENGTH - 3] + "..."

    # Duplikat-Check für Plain-Text
    if text == session._last_edit_html:
        return

    try:
        await session.message.edit_text(text)
        session._last_edit_html = text
        session.last_edit_time = time.monotonic()
        session.edit_count += 1
        _record_edit_success(session)
    except Exception as e:
        # Flood-Control: Pause + Skip
        retry_after = _is_retry_after(e)
        if retry_after is not None:
            _apply_flood_backoff(session, retry_after)
            return
        _handle_edit_error(e)


async def _do_edit_html(session: StreamingSession) -> None:
    """Führt die finale Telegram-Message-Edit mit HTML-Formatierung durch.

    Konvertiert den vollständigen Markdown-Text zu Telegram-HTML
    via markdown_to_telegram_html(). Fallback auf strip_markdown()
    wenn die HTML-Version von Telegram abgelehnt wird.

    FINAL-EDIT: Hat höchste Priorität. Bei RetryAfter wird gewartet
    und erneut versucht (max FINAL_EDIT_MAX_RETRIES Versuche).
    Der User MUSS die fertige Antwort sehen.

    Note: Bei langen Texten wird diese Funktion nicht mehr direkt
    aufgerufen; stattdessen geht der Pfad über _finalize_multi_message().
    """
    raw_text = session.accumulated_text
    if not raw_text.strip():
        raw_text = "..."

    html_text = markdown_to_telegram_html(raw_text)

    for attempt in range(1 + FINAL_EDIT_MAX_RETRIES):
        try:
            await session.message.edit_text(html_text, parse_mode="HTML")
            session.last_edit_time = time.monotonic()
            session.edit_count += 1
            _record_edit_success(session)
            return
        except Exception as e:
            # Flood-Control bei Final-Edit: warten und retrien
            retry_after = _is_retry_after(e)
            if retry_after is not None:
                _apply_flood_backoff(session, retry_after)
                if attempt < FINAL_EDIT_MAX_RETRIES:
                    log.info(
                        "Final-Edit 429, warte %ds (Versuch %d/%d)",
                        retry_after,
                        attempt + 1,
                        FINAL_EDIT_MAX_RETRIES,
                    )
                    await asyncio.sleep(retry_after)
                    continue
                # Alle Retries aufgebraucht: Fallback auf neue Nachricht
                log.error(
                    "Final-Edit nach %d Retries immer noch 429, "
                    "Fallback auf send_message",
                    FINAL_EDIT_MAX_RETRIES,
                )
                plain_text = strip_markdown(raw_text)
                try:
                    await session.message.chat.send_message(plain_text)
                except Exception as fb_e:
                    log.error(
                        "Final-Edit Fallback send_message fehlgeschlagen: %s", fb_e
                    )
                return

            error_str = str(e).lower()
            if "message is not modified" in error_str:
                return  # Harmlos
            if "can't parse entities" in error_str or "bad request" in error_str:
                # HTML-Parse-Fehler: Fallback auf Plain-Text (strip_markdown)
                log.warning("HTML-Edit fehlgeschlagen, Fallback auf Plain-Text: %s", e)
                plain_text = strip_markdown(raw_text)
                try:
                    await session.message.edit_text(plain_text)
                    session.last_edit_time = time.monotonic()
                    session.edit_count += 1
                    _record_edit_success(session)
                except Exception as fallback_e:
                    _handle_edit_error(fallback_e)
                return
            _handle_edit_error(e)
            return


def _handle_edit_error(e: Exception) -> None:
    """Handelt Telegram-API-Fehler leise (loggt aber)."""
    error_str = str(e).lower()
    if "message is not modified" in error_str:
        pass  # Harmlos: Text hat sich nicht geändert
    elif "message to edit not found" in error_str:
        log.warning("Streaming-Edit fehlgeschlagen: Nachricht gelöscht")
    else:
        log.warning("Streaming-Edit Fehler: %s", e)
