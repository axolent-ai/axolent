"""Text Guard: deterministic post-processing filter for LLM output.

Corrects ASCII diacritic substitutions (e.g. ASCII to proper Unicode)
that LLMs frequently produce in non-English languages.

Architecture:
    * guard.py: core engine, language-agnostic
    * models.py: data structures (Issue, WordPair, RuleSet)
    * rules_registry.py: built-in rule sets per language
    * language_detector.py: thin wrapper around domain.language

The domain module is pure: no I/O, no YAML loading, no file access.
Rule sets are Python data structures. YAML files in rules/ are loaded
by the application layer and converted to domain objects.
"""

from domain.text_guard.guard import TextGuard
from domain.text_guard.models import Issue, RuleSet, WordPair
from domain.text_guard.rules_registry import get_builtin_rules, list_languages

__all__ = [
    "Issue",
    "RuleSet",
    "TextGuard",
    "WordPair",
    "get_builtin_rules",
    "list_languages",
]
