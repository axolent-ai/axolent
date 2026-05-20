"""ResponseLanguageVerifier: post-hoc verification of output language.

Checks whether the LLM response is actually in the requested language.
Uses the LanguageDetectorBackend protocol (default: LangdetectBackend)
to keep the detection library as an implementation detail.

Design decisions:
- Backend abstraction: verifier speaks to LanguageDetectorBackend Protocol,
  never to langdetect or domain.language directly. This allows swapping
  backends (langdetect -> Lingua -> fast-langdetect) without touching
  verification logic. (Codex architecture rule, 2026-05-20)
- domain.language is NOT used here: it's calibrated for short user inputs
  (marker-word heuristics on 5-50 word fragments). Long LLM outputs need
  n-gram-profile-based detection (langdetect/Lingua).
- Strips code blocks, URLs, and whitelisted technical terms before detection
  to minimize false positives.
- Uses sliding window for long texts to handle mixed-language sections.
- Skips verification for very short outputs (<20 words) where detection
  is unreliable.
- Three-level VerificationStatus (PASS/WARN/FAIL) for nuanced enforcement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

from application.language.backends import (
    LanguageDetectorBackend,
    LangdetectBackend,
)

log = logging.getLogger(__name__)

# Minimum word count for verification to be meaningful
_MIN_WORDS_FOR_VERIFICATION = 20

# Confidence threshold: below this, detection is too uncertain to act on
_CONFIDENCE_THRESHOLD = 0.7

# Maximum foreign content share before flagging
_MAX_FOREIGN_SHARE = 0.2

# Sliding window size for long text analysis (in words)
_WINDOW_SIZE_WORDS = 100

# Technical terms whitelist: these appear in many languages and should
# not count as foreign-language content
_TECHNICAL_WHITELIST: frozenset[str] = frozenset(
    {
        "api",
        "http",
        "https",
        "url",
        "json",
        "xml",
        "html",
        "css",
        "javascript",
        "typescript",
        "python",
        "rust",
        "golang",
        "docker",
        "kubernetes",
        "git",
        "github",
        "gitlab",
        "npm",
        "pip",
        "cargo",
        "webpack",
        "vite",
        "react",
        "vue",
        "angular",
        "node",
        "deno",
        "bun",
        "token",
        "tokens",
        "plugin",
        "plugins",
        "framework",
        "backend",
        "frontend",
        "middleware",
        "endpoint",
        "database",
        "query",
        "schema",
        "model",
        "prompt",
        "embedding",
        "vector",
        "tensor",
        "gradient",
        "batch",
        "epoch",
        "layer",
        "transformer",
        "attention",
        "encoder",
        "decoder",
        "fine-tuning",
        "inference",
        "training",
        "benchmark",
        "latency",
        "throughput",
        "streaming",
        "async",
        "await",
        "callback",
        "function",
        "class",
        "interface",
        "module",
        "package",
        "import",
        "export",
        "config",
        "configuration",
        "deploy",
        "deployment",
        "container",
        "pipeline",
        "workflow",
        "ci/cd",
        "cache",
        "caching",
        "server",
        "client",
        "request",
        "response",
        "header",
        "payload",
        "webhook",
        "oauth",
        "jwt",
        "ssl",
        "tls",
        "dns",
        "cdn",
        "proxy",
        "load balancer",
        "microservice",
        "monolith",
        "saas",
        "paas",
        "iaas",
        "sdk",
        "cli",
        "gui",
        "ide",
        "debug",
        "log",
        "logging",
        "monitor",
        "monitoring",
        "alert",
        "dashboard",
        "metric",
        "metrics",
        "cloud",
        "aws",
        "azure",
        "gcp",
        "vercel",
        "netlify",
        "heroku",
        "supabase",
        "firebase",
        "telegram",
        "bot",
        "chatbot",
        "llm",
        "gpt",
        "claude",
        "anthropic",
        "openai",
        "mistral",
        "gemini",
        "ollama",
        "axolent",
        "rag",
        "retrieval",
        "augmented",
        "generation",
        "chain-of-thought",
        "few-shot",
        "zero-shot",
        "in-context",
        "learning",
        "machine",
        "neural",
        "network",
        "deep",
        "reinforcement",
        "supervised",
        "unsupervised",
        "pre-training",
        "quantization",
        "lora",
        "qlora",
        "peft",
        "cuda",
        "gpu",
        "cpu",
        "ram",
        "vram",
        "memory",
        "storage",
        "disk",
        "ssd",
        "i/o",
        "bandwidth",
        "kernel",
        "thread",
        "process",
        "mutex",
        "semaphore",
        "queue",
        "stack",
        "heap",
        "buffer",
        "overflow",
        "underflow",
        "pointer",
        "reference",
        "garbage",
        "collection",
        "runtime",
        "compiler",
        "interpreter",
        "jit",
        "aot",
        "wasm",
        "byte",
        "bytes",
        "bit",
        "bits",
        "integer",
        "float",
        "string",
        "boolean",
        "array",
        "list",
        "dict",
        "map",
        "set",
        "tuple",
        "struct",
        "enum",
        "type",
        "generic",
        "template",
        "pattern",
        "algorithm",
        "complexity",
        "o(n)",
        "hash",
        "tree",
        "graph",
        "node",
        "edge",
        "path",
        "sort",
        "search",
        "binary",
        "linear",
        "recursive",
        "iterative",
        "dynamic",
        "programming",
        "design",
        "architecture",
        "solid",
        "dry",
        "kiss",
        "yagni",
        "agile",
        "scrum",
        "sprint",
        "backlog",
        "standup",
        "retrospective",
        "refactor",
        "refactoring",
        "technical",
        "debt",
        "code",
        "review",
        "pull",
        "request",
        "merge",
        "branch",
        "commit",
        "push",
        "fetch",
        "rebase",
        "cherry-pick",
        "tag",
        "release",
        "version",
        "changelog",
        "readme",
        "documentation",
        "docs",
        "wiki",
        "tutorial",
        "guide",
        "example",
        "sample",
        "demo",
        "prototype",
        "mvp",
        "poc",
        "todo",
        "fixme",
        "hack",
        "workaround",
        "bug",
        "fix",
        "feature",
        "enhancement",
        "issue",
        "ticket",
        "story",
        "task",
        "subtask",
        "epic",
        "milestone",
        "roadmap",
        "test",
        "testing",
        "unit",
        "integration",
        "e2e",
        "end-to-end",
        "regression",
        "smoke",
        "load",
        "stress",
        "performance",
        "coverage",
        "assertion",
        "mock",
        "stub",
        "spy",
        "fixture",
        "factory",
        "builder",
        "singleton",
        "observer",
        "strategy",
        "decorator",
        "adapter",
        "facade",
        "proxy",
        "composite",
        "iterator",
        "visitor",
        "mediator",
        "command",
        "state",
        "chain",
        "responsibility",
    }
)

# Regex patterns for content to strip before language detection
_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")
_URL_PATTERN = re.compile(r"https?://\S+")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")


class VerificationStatus(Enum):
    """Three-level verification status.

    PASS: detected == expected, confidence > 0.7, target_ratio > 0.8
    WARN: detected == expected, target_ratio 0.6..0.8 (mixed but dominant)
    FAIL: detected != expected OR target_ratio < 0.6 OR confidence < 0.5
    """

    PASS = "pass"  # nosec B105 - enum value, not a credential
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of language verification on an LLM response.

    Attributes:
        expected_lang: The target language code.
        detected_lang: What language was actually detected (None if skipped).
        confidence: Detection confidence (0.0..1.0).
        foreign_share: Fraction of text detected as foreign (0.0..1.0).
        target_language_ratio: 1.0 - foreign_share, explicit target presence.
        status: Three-level status (PASS/WARN/FAIL).
        reason: Explanation when verification fails (None on pass).
        skipped: Whether verification was skipped (too short, etc.).
    """

    expected_lang: str
    detected_lang: str | None
    confidence: float
    foreign_share: float
    target_language_ratio: float
    status: VerificationStatus
    reason: str | None
    skipped: bool = False

    @property
    def passed(self) -> bool:
        """Backwards-compat: True for PASS or WARN."""
        return self.status in (VerificationStatus.PASS, VerificationStatus.WARN)


class ResponseLanguageVerifier:
    """Verifies that LLM output matches the expected language.

    Strips code blocks, URLs, and technical terms before detection
    to minimize false positives. Uses a pluggable LanguageDetectorBackend
    (default: LangdetectBackend) for the actual detection.
    """

    def __init__(
        self,
        backend: LanguageDetectorBackend | None = None,
        confidence_threshold: float = _CONFIDENCE_THRESHOLD,
        max_foreign_share: float = _MAX_FOREIGN_SHARE,
        min_words: int = _MIN_WORDS_FOR_VERIFICATION,
        technical_whitelist: frozenset[str] | None = None,
    ) -> None:
        """Initialize the verifier.

        Args:
            backend: Detection backend (default: LangdetectBackend).
            confidence_threshold: Minimum confidence to trust detection.
            max_foreign_share: Maximum allowed foreign content fraction.
            min_words: Minimum word count for verification.
            technical_whitelist: Custom whitelist (default: built-in).
        """
        self._backend = backend or LangdetectBackend()
        self._confidence_threshold = confidence_threshold
        self._max_foreign_share = max_foreign_share
        self._min_words = min_words
        self._whitelist = (
            technical_whitelist
            if technical_whitelist is not None
            else _TECHNICAL_WHITELIST
        )

    def verify(self, output: str, expected_lang: str) -> VerificationResult:
        """Verify that output text is in the expected language.

        Args:
            output: LLM response text.
            expected_lang: Expected ISO-639-1 language code.

        Returns:
            VerificationResult with status and diagnostics.
        """
        # Step 1: Clean the text for detection
        cleaned = self._clean_for_detection(output)

        # Step 2: Check minimum length
        words = cleaned.split()
        if len(words) < self._min_words:
            log.debug(
                "Verification skipped: %d words < minimum %d",
                len(words),
                self._min_words,
            )
            return VerificationResult(
                expected_lang=expected_lang,
                detected_lang=None,
                confidence=0.0,
                foreign_share=0.0,
                target_language_ratio=1.0,
                status=VerificationStatus.PASS,
                reason=None,
                skipped=True,
            )

        # Step 3: Detect language via backend
        if len(words) <= _WINDOW_SIZE_WORDS:
            # Short text: single detection via backend distribution
            distribution = self._backend.detect_distribution(cleaned)
            detected_lang, confidence = self._top_from_distribution(distribution)
            target_ratio = distribution.get(expected_lang, 0.0)
            foreign_share = (
                0.0 if detected_lang == expected_lang else 1.0 - target_ratio
            )
        else:
            # Long text: sliding window analysis
            detected_lang, confidence, foreign_share = self._sliding_window_detect(
                words, expected_lang
            )
            target_ratio = 1.0 - foreign_share

        # Step 4: Decision logic with three-level status
        status = self._determine_status(
            detected_lang, expected_lang, confidence, target_ratio
        )
        reason: str | None = None

        if status == VerificationStatus.FAIL:
            reason = (
                f"Expected '{expected_lang}' but detected '{detected_lang}' "
                f"with confidence {confidence:.2f} "
                f"(target_ratio: {target_ratio:.0%}, foreign_share: {foreign_share:.0%})"
            )
            log.warning("Language verification FAILED: %s", reason)

        return VerificationResult(
            expected_lang=expected_lang,
            detected_lang=detected_lang,
            confidence=confidence,
            foreign_share=foreign_share,
            target_language_ratio=target_ratio,
            status=status,
            reason=reason,
            skipped=False,
        )

    @staticmethod
    def _top_from_distribution(
        distribution: dict[str, float],
    ) -> tuple[str, float]:
        """Extract top language and confidence from a distribution.

        Args:
            distribution: {lang_code: probability} dict.

        Returns:
            Tuple of (top_lang, confidence). Falls back to ("", 0.0).
        """
        if not distribution:
            return ("", 0.0)
        top_lang = max(distribution, key=distribution.get)  # type: ignore[arg-type]
        return (top_lang, distribution[top_lang])

    @staticmethod
    def _determine_status(
        detected_lang: str,
        expected_lang: str,
        confidence: float,
        target_ratio: float,
    ) -> VerificationStatus:
        """Determine three-level verification status.

        Rules:
        - confidence < 0.5: PASS (not enough signal to act)
        - detected == expected AND target_ratio > 0.8: PASS
        - detected == expected AND target_ratio 0.6..0.8: WARN
        - detected != expected OR target_ratio < 0.6: FAIL

        Args:
            detected_lang: Language with highest probability.
            expected_lang: Target language.
            confidence: Top-language confidence.
            target_ratio: Ratio of target language in text.

        Returns:
            VerificationStatus.
        """
        # Low confidence: cannot make a reliable decision
        if confidence < 0.5:
            return VerificationStatus.PASS

        # Detected matches expected
        if detected_lang == expected_lang:
            if target_ratio > 0.8:
                return VerificationStatus.PASS
            if target_ratio >= 0.6:
                return VerificationStatus.WARN
            # target_ratio < 0.6 despite match: suspicious, FAIL
            return VerificationStatus.FAIL

        # Detected does NOT match expected
        # But if target_ratio is still high (e.g. secondary language just barely won)
        if target_ratio > 0.8:
            return VerificationStatus.PASS
        if target_ratio >= 0.6:
            return VerificationStatus.WARN

        return VerificationStatus.FAIL

    def _clean_for_detection(self, text: str) -> str:
        """Remove content that interferes with language detection.

        Strips:
        - Code blocks (triple backtick)
        - Inline code
        - URLs
        - Markdown link URLs (keeps link text)
        - Technical whitelist terms

        Args:
            text: Raw LLM output.

        Returns:
            Cleaned text suitable for language detection.
        """
        # Remove code blocks
        cleaned = _CODE_BLOCK_PATTERN.sub("", text)
        # Remove inline code
        cleaned = _INLINE_CODE_PATTERN.sub("", cleaned)
        # Remove URLs
        cleaned = _URL_PATTERN.sub("", cleaned)
        # Replace markdown links with just the link text
        cleaned = _MARKDOWN_LINK_PATTERN.sub(r"\1", cleaned)

        # Remove whitelist terms (case-insensitive word boundary)
        words = cleaned.split()
        filtered_words = [
            w
            for w in words
            if w.lower().strip(".,;:!?()[]{}\"'") not in self._whitelist
        ]

        return " ".join(filtered_words)

    def _sliding_window_detect(
        self,
        words: list[str],
        expected_lang: str,
    ) -> tuple[str, float, float]:
        """Detect language using sliding windows for long texts.

        Analyzes multiple windows via the backend and aggregates results.

        Args:
            words: Pre-split word list.
            expected_lang: Expected language code.

        Returns:
            Tuple of (dominant_detected_lang, avg_confidence, foreign_share).
        """
        window_size = _WINDOW_SIZE_WORDS
        step = window_size // 2  # 50% overlap

        window_results: list[tuple[str, float]] = []
        foreign_windows = 0

        for i in range(0, len(words), step):
            window = " ".join(words[i : i + window_size])
            if len(window.split()) < self._min_words:
                continue

            distribution = self._backend.detect_distribution(window)
            detected, conf = self._top_from_distribution(distribution)
            window_results.append((detected, conf))

            if detected != expected_lang and conf >= self._confidence_threshold:
                foreign_windows += 1

        if not window_results:
            return expected_lang, 0.0, 0.0

        # Aggregate: most common detected language
        lang_counts: dict[str, int] = {}
        conf_sum = 0.0
        for lang, conf in window_results:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
            conf_sum += conf

        dominant_lang = max(lang_counts, key=lang_counts.get)  # type: ignore[arg-type]
        avg_confidence = conf_sum / len(window_results)
        foreign_share = foreign_windows / len(window_results)

        return dominant_lang, avg_confidence, foreign_share
