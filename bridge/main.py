"""Jarvis-LITE Bridge: Entry-Point.

Startet den Telegram-Bot mit hexagonaler Architektur.
Modus B (lokaler CLI-Wrapper, User hat eigene Pro/Max-Subscription).

Laedt Konfiguration, registriert Provider + Handler, startet Long-Polling.
"""

from __future__ import annotations

import logging
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv

load_dotenv()

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from application.provider_router import ProviderRouter
from application.chat_service import set_provider_router
from infrastructure.personality_loader import build_combined_prompt
from infrastructure.providers import (
    ClaudeProvider,
    GeminiProvider,
    MistralVibeProvider,
    OllamaProvider,
    OpenAICodexProvider,
)
from presentation.callbacks import (
    handle_bookmark_delete_callback,
    handle_bookmark_show_callback,
)
from presentation.decorators import ALLOW_ALL_USERS, WHITELIST
from presentation.handlers import (
    handle_bookmarks_command,
    handle_help_command,
    handle_lang_command,
    handle_message,
    handle_new_command,
    handle_reset_command,
    handle_save_command,
    handle_start_command,
    set_system_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("jarvis-bridge")


def _build_provider_router() -> ProviderRouter:
    """Erstellt und konfiguriert den ProviderRouter mit allen Providern.

    Registriert alle bekannten Provider (aktive + Stubs).
    Default-Provider wird aus .env gelesen oder faellt auf 'claude' zurueck.
    """
    providers = {
        "claude": ClaudeProvider(),
        "openai": OpenAICodexProvider(),
        "gemini": GeminiProvider(),
        "mistral": MistralVibeProvider(),
        "ollama": OllamaProvider(),
    }

    default = os.getenv("DEFAULT_PROVIDER", "claude")

    # Validierung: Default muss registriert sein
    if default not in providers:
        log.warning(
            "DEFAULT_PROVIDER='%s' nicht bekannt, falle auf 'claude' zurueck.",
            default,
        )
        default = "claude"

    router = ProviderRouter(providers=providers, default=default)

    # Log welche Provider tatsaechlich verfuegbar sind
    available = router.list_available()
    log.info("Verfuegbare Provider: %s", available if available else ["KEINE!"])

    return router


def main() -> None:
    """Startet den Jarvis-LITE Bridge Bot via long-polling."""
    # Whitelist-Validierung
    if not WHITELIST and not ALLOW_ALL_USERS:
        log.critical(
            "WHITELIST_USER_IDS in .env nicht gesetzt oder leer. "
            "Setze WHITELIST_USER_IDS=12345 oder ALLOW_ALL_USERS=true (nur fuer Dev!)"
        )
        sys.exit(1)

    # Token laden
    token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        log.critical("Kein TELEGRAM_BOT_TOKEN in .env gefunden.")
        sys.exit(1)

    # Provider-Router initialisieren und in ChatService injizieren
    router = _build_provider_router()
    set_provider_router(router)

    # Personality laden und an Handler injizieren
    system_prompt = build_combined_prompt()
    set_system_prompt(system_prompt)

    # Application bauen
    app = Application.builder().token(token).build()

    # Command handlers
    app.add_handler(CommandHandler("start", handle_start_command))
    app.add_handler(CommandHandler("help", handle_help_command))
    app.add_handler(CommandHandler("save", handle_save_command))
    app.add_handler(CommandHandler("bookmarks", handle_bookmarks_command))
    app.add_handler(CommandHandler("reset", handle_reset_command))
    app.add_handler(CommandHandler("new", handle_new_command))
    app.add_handler(CommandHandler("lang", handle_lang_command))

    # Message handler (non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Callback handlers for inline keyboard buttons
    app.add_handler(
        CallbackQueryHandler(handle_bookmark_show_callback, pattern=r"^bm_show:")
    )
    app.add_handler(
        CallbackQueryHandler(handle_bookmark_delete_callback, pattern=r"^bm_del:")
    )

    log.info("Jarvis-LITE Bridge startet, Modus B (Multi-Provider-Router)")
    log.info("Default-Provider: '%s'", router.default)
    log.info(
        "Whitelist aktiv: %s",
        "ja"
        if WHITELIST
        else ("ALLOW_ALL_USERS=true (Dev-Modus!)" if ALLOW_ALL_USERS else "FEHLER"),
    )
    log.info("Bookmarks-Feature aktiv (Reply-basiert via /save)")
    log.info("Conversation-History aktiv (max 20 Turns, /reset zum Loeschen)")
    app.run_polling()


if __name__ == "__main__":
    main()
