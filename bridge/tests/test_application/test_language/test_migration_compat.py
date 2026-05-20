"""Tests that the backward-compatible re-export shim works.

Verifies that the old import paths still resolve correctly after
migrating LanguageContext and LanguageResolver into the
application.language subpackage.
"""


class TestBackwardCompatImports:
    """Verify old import paths still work."""

    def test_import_language_context_from_old_path(self) -> None:
        """from application.language_resolver import LanguageContext works."""
        from application.language_resolver import LanguageContext

        ctx = LanguageContext(
            code="de",
            source="sticky",
            confidence=1.0,
            switched_from=None,
            request_id="compat-test",
        )
        assert ctx.code == "de"

    def test_import_language_resolver_from_old_path(self) -> None:
        """from application.language_resolver import LanguageResolver works."""
        from application.language_resolver import LanguageResolver

        resolver = LanguageResolver(default_lang="en")
        assert resolver._default == "en"

    def test_import_from_new_path(self) -> None:
        """from application.language import ... works."""
        from application.language import LanguageContext

        ctx = LanguageContext(
            code="en",
            source="override",
            confidence=1.0,
            switched_from=None,
            request_id="new-path-test",
        )
        assert ctx.code == "en"

    def test_same_class_identity(self) -> None:
        """Old and new import paths resolve to the same class."""
        from application.language import LanguageContext as NewCtx
        from application.language_resolver import LanguageContext as OldCtx

        assert NewCtx is OldCtx

    def test_from_code_static_method(self) -> None:
        """LanguageResolver.from_code() works from both paths."""
        from application.language import LanguageResolver as NewResolver
        from application.language_resolver import LanguageResolver as OldResolver

        ctx_old = OldResolver.from_code("fr")
        ctx_new = NewResolver.from_code("fr")

        assert ctx_old.code == ctx_new.code == "fr"
        assert type(ctx_old) is type(ctx_new)
