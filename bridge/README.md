# Bridge-Service

Backend von Jarvis-LITE. Telegram-Bot der Claude Code CLI als lokalen Subprozess spawnt (Modus B). Hexagonale Architektur, 113 Tests, UTF-8 durchgängig.

## Architektur (Hexagonal)

```
[Telegram User]
      |
      v
[presentation/handlers.py]   Telegram-spezifisch: Commands, Messages, Callbacks
      |
      v
[application/services]        Use-Cases: chat_service, bookmark_service
      |                \
      v                 v
[domain/]            [infrastructure/]
  Pure Logic           I/O-Adapter
  bookmark.py          claude_cli.py      (Claude Code CLI Subprozess)
  language.py          bookmark_storage.py (JSONL Persistenz)
  conversation.py      conversation_storage.py
  personality.py       audit_log.py       (Audit mit Rotation)
  markdown.py          encoding.py        (UTF-8 Helper)
                       personality_loader.py
```

**Datenfluss:** Telegram-Nachricht kommt rein -> presentation parsed und validiert -> application orchestriert den Use-Case -> domain enthält die Businesslogik -> infrastructure führt I/O aus (CLI-Aufruf, Dateisystem, Logging).

## Verzeichnisse

| Ordner | Inhalt |
|--------|--------|
| `domain/` | Pure Businesslogik. Keine I/O-Imports erlaubt. |
| `application/` | Use-Case-Orchestration (chat_service, bookmark_service) |
| `infrastructure/` | I/O-Adapter: Claude CLI, Storage, Audit, Encoding |
| `presentation/` | Telegram-Handler, Decorators (Whitelist), Rendering |
| `config/` | system_prompt.md, user_constitution.md |
| `data/` | bookmarks.jsonl (Laufzeit-Daten) |
| `logs/` | audit.jsonl (mit Rotation) |
| `tests/` | 113 pytest-Tests |
| `models/` | Datenmodelle |
| `handlers/` | Legacy (wird migriert nach presentation/) |
| `providers/` | Provider-Abstraktion (Phase 1+) |
| `tools/` | Tool-Definitionen (Phase 1+) |

## Setup

### Voraussetzungen

1. Python 3.11+ (3.12 empfohlen)
2. Claude Code CLI installiert und eingeloggt (eigene Pro/Max Subscription)
3. Telegram Bot Token (via @BotFather)

### Installation

```bash
cd bridge
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### .env anlegen

Erstelle eine `.env` Datei im `bridge/` Ordner:

```env
# Pflicht
TELEGRAM_BOT_TOKEN=dein_bot_token_hier
WHITELIST_USER_IDS=123456789

# Optional (nur fuer Entwicklung!)
ALLOW_ALL_USERS=false
```

## .env Variablen

| Variable | Pflicht | Beschreibung |
|----------|---------|--------------|
| `TELEGRAM_BOT_TOKEN` | Ja | Bot-Token von @BotFather |
| `WHITELIST_USER_IDS` | Ja* | Komma-separierte Telegram User IDs |
| `ALLOW_ALL_USERS` | Nein | `true` = jeder darf den Bot nutzen (nur Dev!) |

*Pflicht wenn `ALLOW_ALL_USERS` nicht auf `true` steht.

## Bot starten

```bash
python main.py
```

Erwartete Log-Ausgabe:

```
2026-05-06 10:00:00 [INFO] jarvis-bridge: Jarvis-LITE Bridge startet, Modus B (Claude Code CLI Subprozess)
2026-05-06 10:00:00 [INFO] jarvis-bridge: Whitelist aktiv: ja
2026-05-06 10:00:00 [INFO] jarvis-bridge: Bookmarks-Feature aktiv (Reply-basiert via /save)
2026-05-06 10:00:00 [INFO] jarvis-bridge: Conversation-History aktiv (max 20 Turns, /reset zum Löschen)
```

Der Bot pollt jetzt Telegram. Jede Nachricht an den Bot wird an Claude Code CLI weitergeleitet.

## Telegram-Commands

| Command | Beschreibung |
|---------|--------------|
| Normaler Text | Startet Claude-Anfrage mit Conversation-History |
| `/save` (als Reply) | Bookmark speichern oder entfernen (Toggle) |
| `/bookmarks` | Liste aller gespeicherten Bookmarks |
| `/bookmarks search <begriff>` | Bookmarks durchsuchen |
| `/lang <code>` | Sprache fest setzen (de, en, es, fr, etc.) |
| `/reset` oder `/new` | Konversation und Sprache zurücksetzen |
| `/help` | Commands-Übersicht |
| `/start` | Begrüßung |

## Tests ausführen

```bash
# Alle Tests
python -m pytest

# Mit Verbose-Output (default via pyproject.toml)
python -m pytest -v

# Mit Coverage
python -m pytest --cov

# Einzelnes Modul
python -m pytest tests/test_bookmark.py
```

Aktuell: **113 Tests**, alle grün, Laufzeit ca. 2 Sekunden.

## Architektur-Regeln (nicht verhandelbar)

| Layer | Darf importieren von |
|-------|---------------------|
| `domain/` | Nichts (pure, keine externen Deps) |
| `infrastructure/` | `domain/` |
| `application/` | `domain/`, `infrastructure/` |
| `presentation/` | `domain/`, `application/` |
| `main.py` | Alles (Composition Root) |

**Goldene Regel:** domain/ importiert NIEMALS aus infrastructure/ oder presentation/. Wenn du diese Regel brichst, brechen die Tests.

## Stil-Regeln

1. Kommentare und Dokumentation: Deutsch
2. Code-Identifier (Variablen, Funktionen, Klassen): Englisch
3. Umlaute immer korrekt: ä, ö, ü, ß (nie ae, oe, ue, ss)
4. Keine Gedankenstriche in Outputs
5. Bullets als Punkt (•) oder nummeriert, nie als Bindestrich
6. Type-Hints durchgängig (alle Funktionen, alle Parameter)
7. Docstrings: was rein, was raus, WARUM (nicht WAS der Code tut)
8. Encoding: immer explizit UTF-8 + errors="replace" + ensure_ascii=False

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| `claude: command not found` | Claude Code CLI installieren und einloggen (`claude login`) |
| `WHITELIST_USER_IDS not set` | In `.env` setzen oder `ALLOW_ALL_USERS=true` für Dev |
| Mojibake im Bot-Output | `PYTHONIOENCODING=utf-8` setzen (main.py tut das automatisch) |
| Bot startet nicht | `pip install -e .` wiederholen, `python -c "import main"` zum Testen |
| Tests schlagen fehl | `.venv` aktiv? `pip install -e ".[test]"` ausführen |
| Bookmark wird nicht gespeichert | `/save` muss als Reply auf eine Bot-Nachricht gesendet werden |
| Claude antwortet nicht | CLI testen: `claude "test"` direkt im Terminal ausführen |
