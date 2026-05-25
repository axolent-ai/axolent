"""Axolent Bridge: Entry point.

Starts the Telegram bot with hexagonal architecture.
Mode B (local CLI wrapper, user has own Pro/Max subscription).

Loads configuration, registers providers + handlers, starts long-polling.
"""

from __future__ import annotations

# Runtime type-checking: typeguard 4.x replaces beartype (2026-05-23).
# beartype crashed on PEP 563 Forward References (`from __future__ import
# annotations`). typeguard uses AST-based instrumentation which resolves
# annotations in source-code context, avoiding the get_type_hints() crash.
#
# Two layers:
#   (1) pytest import-hook: typeguard-packages in pyproject.toml instruments
#       ALL functions in domain/application/infrastructure/presentation
#       during tests. Zero production overhead.
#   (2) @typechecked on critical entry-points: runtime checks in production
#       for the ~15 most important functions (constructors, handlers, pipelines).
#
# forward_ref_policy=WARN: unresolvable forward refs produce warnings
# instead of crashes (graceful degradation).

import logging
import os
import re
import sys
from datetime import datetime, timezone

os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Sentry error tracking (initialized early so it catches startup errors)
# ---------------------------------------------------------------------------
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration


_TELEGRAM_BOT_URL_RE = re.compile(r"(https://api\.telegram\.org/bot)[^/]+(/)")


def _redact_sensitive_url(url: str) -> str:
    """Replace Telegram bot tokens embedded in URL paths.

    Telegram bot API URLs have the form
    ``https://api.telegram.org/bot<TOKEN>/<method>``.
    This replaces ``<TOKEN>`` with ``[REDACTED]``.

    Args:
        url: Original URL string.

    Returns:
        URL with bot token replaced, or the original if no match.
    """
    return _TELEGRAM_BOT_URL_RE.sub(r"\1[REDACTED]\2", url)


# FIX-01: Allowlist for Sentry extra/breadcrumb data keys.
# Only these keys survive; everything else is stripped.
# The old blocklist is kept as a second safety net (defense-in-depth).
_SENTRY_EXTRA_ALLOWLIST: frozenset[str] = frozenset(
    {
        "request_id",
        # CFV-03: user_id removed (raw Telegram user ID is PII).
        # Use chat_id_hash for correlation instead.
        "model_id",
        "lang",
        "session_id",
        "chat_id_hash",
    }
)

# Legacy blocklist: second safety net. Even if a key somehow survives
# the allowlist filter, these are explicitly removed.
_SENTRY_EXTRA_BLOCKLIST: frozenset[str] = frozenset(
    {
        "message_text",
        "user_message",
        "user_input",
        "claim",
        "text",
        "prompt",
        "raw_text",
        "response_text",
        "system_prompt",
        "response",
        "body",
    }
)

# FIX-02 (CFV-02): Privacy-by-default exception redaction.
# ALL exception values are redacted. The previous allowlist approach
# (_SENTRY_REDACT_EXCEPTION_TYPES with 5 types) left gaps: any custom
# exception like Exception("user text") or MyAppError("private data")
# would leak raw user text to Sentry. Removed the type-check entirely.


def _sentry_before_send(event, hint):
    """Strip user-controlled text from Sentry events.

    AXOLENT processes user messages; we never want to send raw
    Telegram messages to Sentry. The audit-log layer handles
    user-text auditing separately (privacy-by-design).

    Three hardening layers (Phase D findings):
      1. Allowlist: only approved keys survive in extra/breadcrumb data.
      2. Exception-value redaction: standard exception messages are
         replaced to prevent user-text leakage.
      3. Frame-locals stripping: defense-in-depth alongside
         ``include_local_variables=False``.

    Args:
        event: The Sentry event dict.
        hint: Additional context (usually contains the exception).

    Returns:
        Modified event or None to drop the event entirely.
    """
    # Strip request bodies / message content and redact bot tokens in URLs
    if "request" in event and isinstance(event["request"], dict):
        event["request"].pop("data", None)
        event["request"].pop("query_string", None)
        if "url" in event["request"] and isinstance(event["request"]["url"], str):
            event["request"]["url"] = _redact_sensitive_url(event["request"]["url"])

    # FIX-01: Allowlist + blocklist for extra context
    if "extra" in event and isinstance(event["extra"], dict):
        # Primary: remove anything NOT in the allowlist
        disallowed = [k for k in event["extra"] if k not in _SENTRY_EXTRA_ALLOWLIST]
        for key in disallowed:
            del event["extra"][key]
        # Secondary: blocklist safety net (in case allowlist is accidentally widened)
        for key in _SENTRY_EXTRA_BLOCKLIST:
            event["extra"].pop(key, None)

    # FIX-02: Sanitize exception values for standard exception types
    if "exception" in event and isinstance(event["exception"], dict):
        for exc_val in event["exception"].get("values", []):
            if not isinstance(exc_val, dict):
                continue
            # CFV-02: Redact ALL exception messages (privacy-by-default)
            if "value" in exc_val:
                exc_val["value"] = "<exception message redacted>"
            # FIX-03: Strip frame locals (defense-in-depth)
            st = exc_val.get("stacktrace")
            if isinstance(st, dict):
                for frame in st.get("frames", []):
                    if isinstance(frame, dict):
                        frame.pop("vars", None)

    # FIX-01 (breadcrumbs): Allowlist + blocklist + bot-token redaction
    breadcrumbs = event.get("breadcrumbs", {}).get("values", [])
    for crumb in breadcrumbs:
        if isinstance(crumb.get("data"), dict):
            # Primary: allowlist
            disallowed = [
                k
                for k in crumb["data"]
                if k not in _SENTRY_EXTRA_ALLOWLIST
                and k not in ("url", "status_code", "method", "category", "handler")
            ]
            for key in disallowed:
                del crumb["data"][key]
            # Secondary: blocklist safety net
            for key in _SENTRY_EXTRA_BLOCKLIST:
                crumb["data"].pop(key, None)
            # Redact bot tokens in URLs
            if "url" in crumb["data"] and isinstance(crumb["data"]["url"], str):
                crumb["data"]["url"] = _redact_sensitive_url(crumb["data"]["url"])

    return event


_sentry_dsn = os.getenv("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        # Release tag from git commit hash (short)
        release=os.getenv("AXOLENT_RELEASE", "dev"),
        # Privacy: never send raw user input
        send_default_pii=False,
        # Privacy: never attach local variables from stack frames
        # (could leak user text captured in handler locals)
        include_local_variables=False,
        # Errors only, no performance tracking (saves quota)
        traces_sample_rate=0.0,
        profiles_sample_rate=0.0,
        integrations=[
            LoggingIntegration(
                level=logging.INFO,  # Capture info+ as breadcrumbs
                event_level=logging.ERROR,  # Send errors as events
            ),
            AsyncioIntegration(),
        ],
        # Filter: drop events that contain user-message text
        # (the audit-log layer is responsible for those)
        before_send=_sentry_before_send,
    )
else:
    pass  # Sentry DSN not set, error tracking disabled

from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from pathlib import Path

from application.bookmark_service import BookmarkService
from application.chat_service import ChatService
from application.memory_service import MemoryService
from application.model_registry import ModelRegistry
from application.model_service import ModelService, resolve_alias
from application.provider_router import ProviderRouter
from application.self_awareness_service import SelfAwarenessService
from application.rate_limiter import RateLimiter
from infrastructure.audit_log import write_audit_log
from infrastructure.bookmark_storage import (
    JsonlBookmarkStorageAdapter,
    migrate_legacy_chat_id,
)
from infrastructure.claude_process_pool import ClaudeProcessPool
from infrastructure.memory_storage import MemoryStorage
from infrastructure.sqlite_storage import (
    SqliteBookmarkStorage,
    SqliteConnection,
    SqliteLanguageStorage,
    SqliteMemoryStorage,
    SqliteModelStorage,
    SqliteProfileStorage,
    SqliteRateLimitStorage,
    migrate_jsonl_to_sqlite,
)
from infrastructure.personality_loader import build_combined_prompt
from infrastructure.providers import (
    ClaudeProvider,
    ClaudePersistentProvider,
    GeminiProvider,
    MistralVibeProvider,
    OllamaProvider,
    OpenAICodexProvider,
)
from infrastructure.onboarding_storage import OnboardingStorage
from presentation.callbacks import (
    handle_bookmark_delete_callback,
    handle_bookmark_show_callback,
)
from presentation.onboarding_callbacks import handle_wizard_callback
from presentation.settings_callbacks import handle_settings_callback
from presentation.decorators import ALLOW_ALL_USERS, WHITELIST
from presentation.handlers import (
    handle_bookmarks_command,
    handle_debate_command,
    handle_forget_command,
    handle_help_command,
    handle_lang_callback,
    handle_lang_command,
    handle_memory_command,
    handle_message,
    handle_models_command,
    handle_new_command,
    handle_onboarding_command,
    handle_remember_command,
    handle_reset_command,
    handle_resetmodel_command,
    handle_stop_command,
    handle_save_command,
    handle_setlimit_command,
    handle_setmodel_command,
    handle_settings_command,
    handle_start_command,
    handle_usage_command,
)
from presentation.skill_commands import (
    handle_explain_command,
    handle_import_callback,
    handle_import_command,
    handle_learn_command,
    handle_skill_callback,
    handle_skill_detail_command,
    handle_skill_forget_command,
    handle_skills_command,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Install global secret-redaction filter BEFORE any HTTP calls happen.
# Masks Telegram bot tokens, Sentry DSNs, API keys, Bearer tokens in all
# log records (root + httpx/telegram/httpcore/anthropic/openai loggers).
from infrastructure.log_redaction import install_secret_redaction_filter

install_secret_redaction_filter()

log = logging.getLogger("axolent")

# Deferred Sentry log (logger only available after basicConfig)
if _sentry_dsn:
    log.info("Sentry initialized (environment=%s)", os.getenv("SENTRY_ENVIRONMENT"))
else:
    log.info("Sentry DSN not set, error tracking disabled")

# Flag: whether DEV_MODE is active (only for ALLOW_ALL_USERS safeguard)
AXOLENT_DEV_MODE: bool = os.getenv("AXOLENT_DEV_MODE", "").lower() in (
    "true",
    "1",
    "yes",
)


def validate_allow_all_users() -> None:
    """Validate that ALLOW_ALL_USERS is safely configured.

    Blocks bot startup when ALLOW_ALL_USERS is active without
    AXOLENT_DEV_MODE. Prevents accidentally opening the bot
    to all Telegram users in production.

    Raises:
        SystemExit: If ALLOW_ALL_USERS=true without AXOLENT_DEV_MODE=true.
    """
    if not ALLOW_ALL_USERS:
        return

    if not AXOLENT_DEV_MODE:
        log.critical(
            "DANGER: ALLOW_ALL_USERS is active but AXOLENT_DEV_MODE is not set. "
            "Set AXOLENT_DEV_MODE=true if this is intentional, "
            "otherwise remove ALLOW_ALL_USERS."
        )
        sys.exit(2)

    log.warning("WARNING: ALLOW_ALL_USERS active in DEV_MODE. Whitelist disabled.")
    write_audit_log(
        {
            "event_type": "dev_mode_start",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "allow_all_users": True,
            "axolent_dev_mode": True,
        }
    )


def _migrate_profiles_to_sqlite(profile_storage: SqliteProfileStorage) -> None:
    """Migrate JSONL profiles to SQLite (one-time, idempotent).

    Reads data/user_profiles.jsonl, writes to SQLite,
    renames JSONL to .bak.
    """
    jsonl_path = Path(__file__).resolve().parent / "data" / "user_profiles.jsonl"
    if not jsonl_path.exists():
        return

    # Only migrate if SQLite table is empty
    existing = profile_storage.load_all()
    if existing:
        return

    import json

    from infrastructure.encoding import open_utf8

    migrated = 0
    profiles: dict[int, tuple[int, str]] = {}  # user_id -> (chat_id, profile)
    try:
        with open_utf8(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    uid = entry.get("user_id")
                    profile = entry.get("profile", "normal")
                    chat_id = entry.get("chat_id", 0)
                    if uid is not None:
                        profiles[int(uid)] = (chat_id, profile)
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
    except OSError:
        return

    for uid, (cid, profile) in profiles.items():
        profile_storage.save(uid, cid, profile)
        migrated += 1

    if migrated > 0:
        bak_path = jsonl_path.with_suffix(".jsonl.bak")
        jsonl_path.rename(bak_path)
        log.info(
            "Profile migration: %d profiles migrated from JSONL to SQLite",
            migrated,
        )


def _build_provider_router(process_pool: ClaudeProcessPool) -> ProviderRouter:
    """Create and configure the ProviderRouter with all providers.

    Registers all known providers (active + stubs).
    Default provider: claude_persistent (R04, persistent stdin pipe).
    Fallback: claude (legacy, individual subprocesses).

    Args:
        process_pool: ClaudeProcessPool for the PersistentProvider.
    """
    persistent_provider = ClaudePersistentProvider(process_pool=process_pool)

    providers = {
        "claude_persistent": persistent_provider,
        "claude": ClaudeProvider(),
        "openai": OpenAICodexProvider(),
        "gemini": GeminiProvider(),
        "mistral": MistralVibeProvider(),
        "ollama_local": OllamaProvider(),
    }

    default = os.getenv("DEFAULT_PROVIDER", "claude_persistent")

    # Validation: default must be a registered provider
    if default not in providers:
        log.warning(
            "DEFAULT_PROVIDER='%s' not recognized, falling back to 'claude_persistent'.",
            default,
        )
        default = "claude_persistent"

    router = ProviderRouter(providers=providers, default=default)

    # Log which providers are actually available
    available = router.list_available()
    log.info("Available providers: %s", available if available else ["NONE!"])

    return router


def main() -> None:
    """Start the Axolent Bridge Bot via long-polling."""
    # Ollama auto-start (best-effort, non-blocking on failure)
    from application.ollama_service import ensure_ollama_running

    ensure_ollama_running()

    # Whitelist validation
    if not WHITELIST and not ALLOW_ALL_USERS:
        log.critical(
            "WHITELIST_USER_IDS not set or empty in .env. "
            "Set WHITELIST_USER_IDS=12345 or ALLOW_ALL_USERS=true (dev only!)"
        )
        sys.exit(1)

    # C-1: ALLOW_ALL_USERS-Safeguard
    validate_allow_all_users()

    # Load token
    token: str | None = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        log.critical("No TELEGRAM_BOT_TOKEN found in .env.")
        sys.exit(1)

    # Legacy bookmark migration (backfill chat_id, JSONL)
    migrated_count = migrate_legacy_chat_id()
    if migrated_count:
        log.info(
            "Bookmark migration: %d entries backfilled with chat_id", migrated_count
        )

    # C-4: Initialize SQLite storage
    bridge_root = Path(__file__).resolve().parent
    use_sqlite = os.getenv("USE_SQLITE_STORAGE", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    if use_sqlite:
        import time as _time

        _t0 = _time.monotonic()

        sqlite_conn = SqliteConnection(bridge_root / "data" / "axolent.db")

        # JSONL -> SQLite migration (idempotent, first run only)
        migration_stats = migrate_jsonl_to_sqlite(sqlite_conn, bridge_root / "data")
        _duration = _time.monotonic() - _t0

        if migration_stats:
            total_migrated = sum(migration_stats.values())
            log.info(
                "C-4 migration: %d entries migrated in %.2fs %s",
                total_migrated,
                _duration,
                migration_stats,
            )
            write_audit_log(
                {
                    "event_type": "storage_migration",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "migrated_bookmarks": migration_stats.get("bookmarks", 0),
                    "migrated_memory_episodic": migration_stats.get(
                        "memory_episodic", 0
                    ),
                    "migrated_memory_semantic": migration_stats.get(
                        "memory_semantic", 0
                    ),
                    "migrated_memory_procedural": migration_stats.get(
                        "memory_procedural", 0
                    ),
                    "duration_seconds": round(_duration, 3),
                }
            )

        # BookmarkService with SQLite backend (constructor injection)
        bookmark_svc = BookmarkService(storage=SqliteBookmarkStorage(sqlite_conn))

        # Memory-Storage: SQLite
        memory_storage: MemoryStorage | SqliteMemoryStorage = SqliteMemoryStorage(
            sqlite_conn
        )
        log.info("Storage backend: SQLite (WAL mode, FTS5 active)")
    else:
        memory_storage = MemoryStorage(data_dir=bridge_root / "data")

        # BookmarkService with JSONL adapter (constructor injection)
        bookmark_svc = BookmarkService(storage=JsonlBookmarkStorageAdapter())
        log.info("Storage backend: JSONL (legacy mode)")

    # R04: Initialize process pool (for persistent Claude subprocesses)
    process_pool = ClaudeProcessPool()

    # Initialize provider router (with process pool)
    router = _build_provider_router(process_pool)

    # Initialize Trinity memory
    memory_svc = MemoryService(storage=memory_storage)

    # R18 Phase 2a: Load SlotConfigs (needed for ModelService slot defaults)
    from application.task_router import TaskRouter, load_slot_configs

    slot_configs = load_slot_configs()

    # Extract slot defaults: slot_name -> resolved model_id
    _slot_defaults: dict[str, str] = {}
    for cfg in slot_configs:
        if cfg.default_model:
            resolved = resolve_alias(cfg.default_model)
            if resolved:
                _slot_defaults[cfg.slot.value] = resolved

    # R18 Phase 1: Initialize model service (with slot defaults for implicit reset)
    model_svc: ModelService | None = None
    if use_sqlite:
        model_storage = SqliteModelStorage(sqlite_conn)
        model_svc = ModelService(storage=model_storage, slot_defaults=_slot_defaults)
        log.info("R18: Model service initialized (user model override active)")

    # R18 Phase 2a: Initialize TaskRouter
    task_router = TaskRouter(slot_configs=slot_configs, model_service=model_svc)
    log.info("R18 Phase 2a: TaskRouter initialized (%d slots)", len(slot_configs))

    # Initialize onboarding storage (setup wizard)
    onboarding_storage: OnboardingStorage | None = None
    if use_sqlite:
        onboarding_storage = OnboardingStorage(sqlite_conn)
        # Migrate: mark existing users as onboarded
        migrated_onboarding = onboarding_storage.migrate_existing_users(sqlite_conn)
        if migrated_onboarding > 0:
            log.info(
                "Onboarding: %d existing users migrated as onboarded",
                migrated_onboarding,
            )

    # Initialize SelfAwarenessService (Phase 3: extracted from ChatService)
    self_awareness_svc = SelfAwarenessService(
        model_service=model_svc,
        task_router=task_router,
        model_registry=ModelRegistry(),
    )

    # Initialize ProactiveTriggerService (time-awareness + nudges)
    from application.proactive_trigger_service import ProactiveTriggerService

    proactive_trigger_svc = ProactiveTriggerService()

    # Initialize StyleAdaptionService (P3: adaptive communication style)
    from application.style_adaption_service import StyleAdaptionService

    style_adaption_svc = StyleAdaptionService()

    # Initialize FallbackResolver (provider failover for non-streaming)
    from application.fallback_resolver import (
        FallbackResolver,
        load_fallback_config_from_env,
    )

    fallback_config = load_fallback_config_from_env()
    fallback_resolver = FallbackResolver(
        provider_router=router,
        fallback_chains=fallback_config["fallback_chains"],
        timeout_seconds=fallback_config["timeout_seconds"],
        user_notice_threshold=fallback_config["user_notice_threshold"],
    )

    # LCP: Initialize LanguageEnforcement (verify + repair pipeline)
    # Codex Finding 5: audit_log injected as adapter (hexagonal rule).
    from application.language.enforcement import LanguageEnforcement

    language_enforcement = LanguageEnforcement(
        provider_router=router,
        audit_log=write_audit_log,
    )
    log.info("LCP: LanguageEnforcement initialized (Verifier + RepairService)")

    # Skill-Compression: Initialize full pipeline (Review Fix SC-01)
    from application.skill_compression.hypothesis_storage import HypothesisStorage
    from application.skill_compression.pattern_judge import PatternJudge
    from application.skill_compression.privacy.privacy_pipeline import PrivacyPipeline
    from application.skill_compression.skill_explainer import SkillExplainer
    from application.skill_compression.skill_learning_service import (
        SkillLearningService,
    )
    from application.skill_compression.skill_matcher import SkillMatcher

    skill_matcher = None
    hypothesis_storage = None
    skill_explainer = None
    import_orchestrator = None
    skill_learning_service = None

    if use_sqlite:
        hypothesis_storage = HypothesisStorage(sqlite_conn)
        hypothesis_storage.init_schema()

        privacy_pipeline = PrivacyPipeline()
        pattern_judge = PatternJudge(privacy_pipeline=privacy_pipeline)

        skill_matcher = SkillMatcher(
            storage=hypothesis_storage,
            pattern_judge=pattern_judge,
        )
        skill_explainer = SkillExplainer(hypothesis_storage)
        skill_learning_service = SkillLearningService(
            storage=hypothesis_storage,
            privacy_pipeline=privacy_pipeline,
        )

        from application.skill_compression.conversation_import.orchestrator import (
            ImportOrchestrator,
        )

        import_orchestrator = ImportOrchestrator(hypothesis_storage)
        import_orchestrator.init_schema()

        log.info(
            "Skill-Compression: pipeline initialized "
            "(HypothesisStorage + PrivacyPipeline + PatternJudge + "
            "SkillMatcher + SkillExplainer + ImportOrchestrator)"
        )

    # Create ChatService with constructor injection
    chat_service = ChatService(
        provider_router=router,
        memory_service=memory_svc,
        model_service=model_svc,
        task_router=task_router,
        self_awareness_service=self_awareness_svc,
        proactive_trigger_service=proactive_trigger_svc,
        style_adaption_service=style_adaption_svc,
        fallback_resolver=fallback_resolver,
        language_enforcement=language_enforcement,
        skill_matcher=skill_matcher,
    )

    log.info("Trinity memory system initialized (auto-loading active)")

    # C-2: Initialize rate limiter (with SQLite profile + counter storage)
    if use_sqlite:
        profile_storage = SqliteProfileStorage(sqlite_conn)
        rate_limit_storage = SqliteRateLimitStorage(sqlite_conn)
        rate_limiter = RateLimiter(
            profile_storage=profile_storage,
            rate_limit_storage=rate_limit_storage,
        )

        # JSONL profiles -> SQLite migration (one-time)
        _migrate_profiles_to_sqlite(profile_storage)

        # Language persistence: load from SQLite on startup
        from infrastructure.conversation_storage import init_language_storage

        lang_storage = SqliteLanguageStorage(sqlite_conn)
        init_language_storage(lang_storage)
    else:
        rate_limiter = RateLimiter()

    # Load personality
    system_prompt = build_combined_prompt()

    # Phase 0, Commit 2: Initialize Execution Kernel (ContextKernel)
    from application.execution import ContextKernel
    from application.language.audit import DetectionAuditLogger
    from application.language_resolver import LanguageResolver

    detection_audit_logger = DetectionAuditLogger()
    context_kernel = ContextKernel.create_default(
        language_resolver=LanguageResolver(audit_logger=detection_audit_logger),
    )
    log.info("Phase 0: ContextKernel initialized (Language + Time + Channel resolvers)")
    log.info("LCP: DetectionAuditLogger active (structured language audit events)")

    # Build application
    app = Application.builder().token(token).build()

    # Share all services via bot_data (for handler access)
    app.bot_data["chat_service"] = chat_service
    app.bot_data["bookmark_service"] = bookmark_svc
    app.bot_data["system_prompt"] = system_prompt
    app.bot_data["memory_service"] = memory_svc
    app.bot_data["process_pool"] = process_pool
    app.bot_data["persistent_provider"] = router.providers.get("claude_persistent")
    app.bot_data["rate_limiter"] = rate_limiter
    app.bot_data["context_kernel"] = context_kernel
    app.bot_data["language_enforcement"] = language_enforcement
    if model_svc is not None:
        app.bot_data["model_service"] = model_svc
    app.bot_data["task_router"] = task_router
    if use_sqlite:
        app.bot_data["sqlite_conn"] = sqlite_conn
    if onboarding_storage is not None:
        app.bot_data["onboarding_storage"] = onboarding_storage

    # Skill-Compression bot_data entries (Review Fix SC-01)
    if hypothesis_storage is not None:
        app.bot_data["hypothesis_storage"] = hypothesis_storage
    if skill_explainer is not None:
        app.bot_data["skill_explainer"] = skill_explainer
    if import_orchestrator is not None:
        app.bot_data["import_orchestrator"] = import_orchestrator
    if skill_learning_service is not None:
        app.bot_data["skill_learning_service"] = skill_learning_service

    # Lifecycle hooks: start/stop ProcessPool
    async def post_init(application: Application) -> None:
        """Start the ProcessPool cleanup task after app init."""
        await process_pool.start()
        log.info("R04: ClaudeProcessPool started (persistent stdin pipe active)")

    async def post_shutdown(application: Application) -> None:
        """Graceful shutdown: terminate subprocesses, close SQLite connection."""
        await process_pool.shutdown()
        log.info("R04: ClaudeProcessPool shut down")
        # Close SQLite connection cleanly
        conn = application.bot_data.get("sqlite_conn")
        if conn is not None:
            conn.close()
            log.info("SQLite connection closed")

    app.post_init = post_init
    app.post_shutdown = post_shutdown

    # Command handlers
    app.add_handler(CommandHandler("start", handle_start_command))
    app.add_handler(CommandHandler("help", handle_help_command))
    app.add_handler(CommandHandler("save", handle_save_command))
    app.add_handler(CommandHandler("bookmarks", handle_bookmarks_command))
    app.add_handler(CommandHandler("reset", handle_reset_command))
    app.add_handler(CommandHandler("stop", handle_stop_command))
    app.add_handler(CommandHandler("new", handle_new_command))
    app.add_handler(CommandHandler("lang", handle_lang_command))
    app.add_handler(CommandHandler("remember", handle_remember_command))
    app.add_handler(CommandHandler("memory", handle_memory_command))
    app.add_handler(CommandHandler("forget", handle_forget_command))
    app.add_handler(CommandHandler("usage", handle_usage_command))
    app.add_handler(CommandHandler("setlimit", handle_setlimit_command))
    app.add_handler(CommandHandler("setmodel", handle_setmodel_command))
    app.add_handler(CommandHandler("resetmodel", handle_resetmodel_command))
    app.add_handler(CommandHandler("models", handle_models_command))
    app.add_handler(CommandHandler("settings", handle_settings_command))
    app.add_handler(CommandHandler("debate", handle_debate_command))
    app.add_handler(CommandHandler("onboarding", handle_onboarding_command))

    # Skill-Compression command handlers (Review Fix SC-01)
    app.add_handler(CommandHandler("skills", handle_skills_command))
    app.add_handler(CommandHandler("skill", handle_skill_detail_command))
    app.add_handler(CommandHandler("skillforget", handle_skill_forget_command))
    app.add_handler(CommandHandler("learn", handle_learn_command))
    app.add_handler(CommandHandler("explain", handle_explain_command))
    app.add_handler(CommandHandler("import", handle_import_command))

    # Message handler (non-command text)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Callback handlers for inline keyboard buttons
    app.add_handler(
        CallbackQueryHandler(handle_bookmark_show_callback, pattern=r"^bm_show:")
    )
    app.add_handler(
        CallbackQueryHandler(handle_bookmark_delete_callback, pattern=r"^bm_del:")
    )
    app.add_handler(
        CallbackQueryHandler(handle_settings_callback, pattern=r"^settings_")
    )
    app.add_handler(CallbackQueryHandler(handle_wizard_callback, pattern=r"^wizard_"))
    app.add_handler(CallbackQueryHandler(handle_lang_callback, pattern=r"^lang_set:"))

    # Skill-Compression callback handlers (Review Fix SC-01)
    app.add_handler(CallbackQueryHandler(handle_skill_callback, pattern=r"^skill_"))
    app.add_handler(CallbackQueryHandler(handle_import_callback, pattern=r"^import_"))

    log.info("Axolent Bridge starting, Mode B (R04: Persistent Pipe + Streaming)")
    log.info("Default-Provider: '%s'", router.default)
    log.info(
        "Whitelist active: %s",
        "yes"
        if WHITELIST
        else ("ALLOW_ALL_USERS=true (dev mode!)" if ALLOW_ALL_USERS else "ERROR"),
    )
    log.info("Bookmarks feature active (reply-based via /save)")
    log.info("Trinity memory active (/remember /memory /forget)")
    log.info("Conversation history active (max 20 turns, /reset to clear)")
    app.run_polling()


if __name__ == "__main__":
    main()
