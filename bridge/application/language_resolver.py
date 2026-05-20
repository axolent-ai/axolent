"""LanguageResolver: backward-compatibility re-export shim.

.. deprecated:: Language Control Plane refactor
    The canonical implementations now live in:
    - application.language.context.LanguageContext
    - application.language.resolver.LanguageResolver

    This module re-exports both so that existing imports
    (``from application.language_resolver import ...``) continue
    to work without modification. New code should import from
    ``application.language`` directly.
"""

from application.language.context import LanguageContext
from application.language.resolver import LanguageResolver

__all__ = ["LanguageContext", "LanguageResolver"]
