"""Static Model Registry: laedt Modell-Metadaten aus YAML.

Phase 2b: Foundation fuer Phase 2 (TaskRouter) und Phase 2c (Dynamic Scraping).
Konsolidiert alle Modell-Informationen in einer zentralen, datengetriebenen Registry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

# Default YAML path relative to this module
_DEFAULT_YAML = Path(__file__).parent.parent / "config" / "models.yaml"


@dataclass(frozen=True)
class ModelMetadata:
    """Immutable metadata for a single model."""

    id: str
    display_name: str
    provider: str
    aliases: tuple[str, ...]
    context_window: int
    pricing_input_per_mtok: float
    pricing_output_per_mtok: float
    scores: dict[str, float]  # coding, reasoning, knowledge, speed
    supports_thinking: bool
    supports_effort: bool
    is_open_source: bool

    def get_score(self, dimension: str) -> Optional[float]:
        """Return score for a dimension, or None if not present."""
        return self.scores.get(dimension)


class ModelRegistry:
    """Loads models from YAML. Lookup by ID or alias.

    Thread-safe after construction (read-only after _load).
    """

    def __init__(self, yaml_path: Path | str | None = None) -> None:
        if yaml_path is None:
            yaml_path = _DEFAULT_YAML
        self._path = Path(yaml_path)
        self._models: dict[str, ModelMetadata] = {}
        self._alias_index: dict[str, str] = {}  # alias -> model_id
        self._load()

    def _load(self) -> None:
        """Parse YAML and build internal indices."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log.error("Model registry YAML not found: %s", self._path)
            raise
        except OSError as exc:
            log.error("Failed to read model registry YAML: %s", exc)
            raise

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            log.error("Invalid YAML in model registry: %s", exc)
            raise

        if not isinstance(data, dict) or "models" not in data:
            msg = f"Invalid model registry format: expected top-level 'models' key in {self._path}"
            raise ValueError(msg)

        models_list = data["models"]
        if not isinstance(models_list, list):
            msg = f"Invalid model registry format: 'models' must be a list in {self._path}"
            raise ValueError(msg)

        seen_aliases: dict[str, str] = {}  # alias -> first model_id (for dup check)

        for entry in models_list:
            try:
                aliases_raw = entry.get("aliases", [])
                aliases = tuple(str(a).lower().strip() for a in aliases_raw)

                meta = ModelMetadata(
                    id=str(entry["id"]),
                    display_name=str(entry["display_name"]),
                    provider=str(entry["provider"]),
                    aliases=aliases,
                    context_window=int(entry["context_window"]),
                    pricing_input_per_mtok=float(entry["pricing_input_per_mtok"]),
                    pricing_output_per_mtok=float(entry["pricing_output_per_mtok"]),
                    scores={k: float(v) for k, v in entry.get("scores", {}).items()},
                    supports_thinking=bool(entry.get("supports_thinking", False)),
                    supports_effort=bool(entry.get("supports_effort", False)),
                    is_open_source=bool(entry.get("is_open_source", False)),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning(
                    "Skipping invalid model entry %s: %s",
                    entry.get("id", "<unknown>"),
                    exc,
                )
                continue

            if meta.id in self._models:
                log.warning("Duplicate model ID '%s', keeping first", meta.id)
                continue

            self._models[meta.id] = meta

            # Build alias index
            for alias in aliases:
                if alias in seen_aliases:
                    log.warning(
                        "Duplicate alias '%s' (models: '%s' and '%s'), keeping first",
                        alias,
                        seen_aliases[alias],
                        meta.id,
                    )
                    continue
                seen_aliases[alias] = meta.id
                self._alias_index[alias] = meta.id

            # Also index by model ID itself (lowercase) for direct lookups
            lower_id = meta.id.lower()
            if lower_id not in self._alias_index:
                self._alias_index[lower_id] = meta.id

        log.info(
            "ModelRegistry loaded: %d models, %d aliases from %s",
            len(self._models),
            len(self._alias_index),
            self._path,
        )

    def get(self, id_or_alias: str) -> Optional[ModelMetadata]:
        """Resolve an ID or alias to ModelMetadata. Case-insensitive."""
        key = id_or_alias.lower().strip()
        model_id = self._alias_index.get(key)
        if model_id is None:
            return None
        return self._models.get(model_id)

    def resolve_id(self, id_or_alias: str) -> Optional[str]:
        """Resolve an alias or ID to the canonical model ID. Returns None if unknown."""
        key = id_or_alias.lower().strip()
        return self._alias_index.get(key)

    def all(self) -> list[ModelMetadata]:
        """Return all registered models."""
        return list(self._models.values())

    def all_ids(self) -> set[str]:
        """Return set of all canonical model IDs."""
        return set(self._models.keys())

    def all_aliases(self) -> dict[str, str]:
        """Return alias -> model_id mapping (excludes ID-as-alias entries)."""
        result: dict[str, str] = {}
        for meta in self._models.values():
            for alias in meta.aliases:
                result[alias] = meta.id
        return result

    def for_provider(self, provider: str) -> list[ModelMetadata]:
        """Return all models for a given provider."""
        lower = provider.lower().strip()
        return [m for m in self._models.values() if m.provider == lower]

    def best_for_dimension(
        self,
        dimension: str,
        providers: list[str] | None = None,
    ) -> Optional[ModelMetadata]:
        """Return model with highest score in the given dimension.

        Args:
            dimension: Score key (coding, reasoning, knowledge, speed).
            providers: Optional filter to specific providers.

        Returns:
            ModelMetadata with highest score, or None if no model has the dimension.
        """
        candidates = self._models.values()
        if providers:
            lower_providers = {p.lower().strip() for p in providers}
            candidates = [m for m in candidates if m.provider in lower_providers]

        best: Optional[ModelMetadata] = None
        best_score: float = -1.0

        for model in candidates:
            score = model.get_score(dimension)
            if score is not None and score > best_score:
                best_score = score
                best = model

        return best

    def get_display_name(self, model_id: str) -> str:
        """Return display name for a model ID, or the ID itself if unknown."""
        meta = self._models.get(model_id)
        if meta is not None:
            return meta.display_name
        return model_id


# Module-level singleton (lazy)
_registry: Optional[ModelRegistry] = None


def get_registry(yaml_path: Path | str | None = None) -> ModelRegistry:
    """Get or create the module-level ModelRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(yaml_path)
    return _registry


def reset_registry() -> None:
    """Reset the singleton (for testing)."""
    global _registry
    _registry = None
