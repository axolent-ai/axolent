"""Data structures for the Text Guard module.

Pure value objects, no logic, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WordPair:
    """A single ASCII-to-correct mapping.

    Attributes:
        ascii_form: The incorrect ASCII form (e.g. "fuer").
        correct_form: The correct Unicode form (e.g. "fuer" with umlaut).
        case_sensitive: If True, only match exact case. Default: False.
    """

    ascii_form: str
    correct_form: str
    case_sensitive: bool = False


@dataclass(frozen=True, slots=True)
class Issue:
    """A detected diacritic issue in text.

    Attributes:
        line: 1-based line number.
        column: 0-based column offset within the line.
        ascii_form: The ASCII form found.
        correct_form: The suggested correction.
        excerpt: Context around the issue (the full line or a substring).
    """

    line: int
    column: int
    ascii_form: str
    correct_form: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Complete rule set for one language.

    Attributes:
        language: ISO 639-1 code (e.g. "de", "fr").
        word_pairs: Ordered list of word pair corrections.
        loan_word_whitelist: Words that must NOT be corrected
            (e.g. "queue", "blue" for German rules).
        code_block_skip: Whether to skip content inside code blocks.
    """

    language: str
    word_pairs: tuple[WordPair, ...] = ()
    loan_word_whitelist: frozenset[str] = field(default_factory=frozenset)
    code_block_skip: bool = True
