"""Conversation Import sub-package for Skill-Compression.

Provides parsers for importing historical conversations from external
sources (ChatGPT, Claude, Markdown notes, plain text) and an orchestrator
that drives dry-run previews and actual imports.

HC-SC-16: Strictly opt-in, dry-run first, progress display, never periodic.
HC-IMPORT-1: All imported hypotheses start as 'suggested', never 'active'.
HC-IMPORT-2: Raw input text never becomes hypothesis claim.
HC-IMPORT-3: Source deletable via cascade delete.
"""
