"""i18n package: JSON-based internationalization for Axolent.

Public API:
    t(key, lang, **kwargs)           - Main translation function
    get_supported_languages()        - Returns list of language metadata
    is_supported(lang)               - Check if language code is supported
"""

from i18n.domain.i18n import (
    get_supported_languages,
    is_supported,
    t,
)

__all__ = ["t", "get_supported_languages", "is_supported"]
