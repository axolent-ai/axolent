# AXOLENT AI Defensive Publication

**Date:** 2026-05-20 (updated 2026-05-21)
**Author:** AXOLENT AI Project
**Purpose:** Establish prior art for AXOLENT's distinctive architectural choices across six architecture axes, ahead of public release under AGPL-3.0.

---

## 1. Mode B Architecture

AXOLENT implements a local-first AI assistant architecture designated "Mode B": the application runs as a subprocess wrapper on the user's machine, interfacing with the user's own Claude Pro/Max subscription via Claude Code CLI. There is no intermediary server, no token relay, no OAuth hijacking.

The key architectural boundary: AXOLENT is an interface layer. It does not host models, does not proxy API keys to third parties, and does not store conversations on external infrastructure. All data remains local.

This is architecturally distinct from SaaS wrappers (Mode A) that relay user credentials through a cloud service.

**Code reference:** `bridge/main.py` (composition root), `bridge/infrastructure/claude_process_pool.py` (subprocess management)

---

## 2. Language Control Plane

The Language Control Plane (LCP) is AXOLENT's primary architectural innovation in the language-handling domain. It treats language correctness as a system-level guarantee rather than a model-level hope.

As of May 2026, a systematic comparison of all major AI assistants (ChatGPT/OpenAI, Gemini/Google, Claude/Anthropic, Le Chat/Mistral, Perplexity, Pi/Inflection, Microsoft Copilot) revealed that none implements an architecturally isolated language subsystem with runtime verification, repair, and streaming drift recovery. All rely exclusively on prompt-level instructions and model adherence without post-hoc validation.

The LCP consists of five components:

### 2.1 Output Verifier

**Concept:** Every model response undergoes post-hoc language verification before delivery to the user. The verifier detects the language of the generated output using n-gram-profile-based detection and compares it against the declared target language from the request context.

**Technical implementation:**
- Three-level verdict: PASS (correct language, high confidence), WARN (uncertain detection or borderline), FAIL (wrong language detected with high confidence)
- Pre-processing pipeline strips code blocks, URLs, and whitelisted technical terms before detection to prevent false positives from code-heavy responses
- Sliding-window analysis for long texts to handle mixed-language sections accurately
- Minimum word threshold (20 words) below which verification is skipped (insufficient signal)
- Confidence threshold (0.7) below which no enforcement action is taken
- Backend abstraction via `LanguageDetectorBackend` Protocol: the verifier never imports detection libraries directly

The `OrchestratedDetection` result includes a `had_dissent` boolean field that records whether the consulted detection backends disagreed on the detected language. When dissent occurs, the higher-confidence backend wins and a 0.20 reliability penalty is applied. This dissent-tracking mechanism provides an audit signal for ambiguous detections (e.g. German vs. Dutch) that would otherwise be invisible.

**Differentiation from competition:** No other major AI assistant performs runtime language verification on generated output. All competitors deliver the model's response without checking whether it matches the requested language. Language drift is invisible until the user notices.

**Code reference:** `bridge/application/language/verifier.py`, `bridge/application/language/orchestrator.py` (OrchestratedDetection with `had_dissent`)

### 2.2 Repair Loop

**Concept:** When the Output Verifier returns FAIL, the system automatically re-queries the same LLM provider with a reinforced language contract. The original prompt is augmented with an explicit, stronger language instruction derived from the `LanguageContract` builder.

**Technical implementation:**
- Hard-capped at 1 repair attempt (configurable up to 2, never infinite)
- Only triggers when the model's adherence profile has `repair_enabled=True`
- For outputs exceeding 5000 characters: sample-verification only, no full rewrite (protects latency and token budget)
- Emits audit events for monitoring repair frequency per model and per language
- `RepairResult` frozen dataclass captures: original output, repaired output, whether repair was attempted, whether it succeeded, and latency cost

**Differentiation from competition:** No other major AI assistant implements automatic re-query on language verification failure. If the model responds in the wrong language, competitors deliver it unchanged. The user must manually re-prompt.

**Code reference:** `bridge/application/language/repair_service.py`

### 2.3 Stream Guard

**Concept:** For streaming responses (token-by-token delivery), the Stream Guard monitors accumulated text between 200-400 characters and signals an early abort if the model is clearly generating in the wrong language. This prevents streaming an entire wrong-language response that would then need full repair.

**Technical implementation:**
- Single check point between 200-400 accumulated characters (not before, where signal is insufficient)
- Very high confidence threshold (0.85) for early abort, deliberately conservative to minimize false-positive stream interruptions
- Self-calibration mechanism: tracks consecutive false positives and automatically disables after 3 consecutive FP or when the session-level FP rate exceeds 5%. This mechanism is wired into the production streaming path via `StreamGuardStats`, which persists calibration state across streaming sessions and feeds `report_final_outcome()` after final verification of each completed stream.
- Per-session state tracking via `StreamGuardState` dataclass
- Only active for models where `stream_guard_enabled=True` in the adherence profile

**Differentiation from competition:** No other major AI assistant implements mid-stream language drift detection. Competitors stream the full response regardless of language correctness. Users see wrong-language text arrive token-by-token with no intervention.

**Code reference:** `bridge/application/language/stream_guard.py`

### 2.4 Immutable Language Context

**Concept:** The target language for each request is resolved once (via a priority cascade: explicit override, sticky preference, detected from input, default) and frozen into an immutable dataclass. No downstream component can mutate the language decision mid-pipeline.

**Technical implementation:**
- `LanguageContext` is a `frozen=True, slots=True` dataclass containing: ISO-639-1 code, resolution source, confidence score, smart-switch history, and request_id for audit correlation
- Priority cascade: override (confidence 1.0) > sticky (confidence 1.0) > detected (variable confidence, must exceed 0.7 for smart-switch) > default ("de", confidence 1.0)
- `switched_from` field tracks implicit language changes for audit and UX signaling
- All consumers (ChatService, DebateOrchestrator, InstructionCompiler, StreamingHandler) receive the same LanguageContext object; none may resolve language independently
- The `detection_distribution` field uses `types.MappingProxyType` rather than a plain dict. This enforces true immutability at the container level: even though the dataclass is frozen (attribute reassignment blocked), a plain dict would still allow `.update()` or key assignment on the distribution contents. `MappingProxyType` wraps the dict in a read-only view, making both the attribute binding and the container contents immutable. This is enforced via `__post_init__`, which auto-converts any plain dict passed by callers.
- `with_request_id()` helper method returns a copy with a new request_id while preserving all Phase 2 detection metadata (detection_distribution, reliability_score, confidence_history, detection_tier, text_length_bucket, backends_consulted). This prevents the Execution Kernel adapter from silently dropping metadata fields when synchronizing request IDs.

**Differentiation from competition:** Competitors infer language mutably from conversation context. When code, English error messages, or long technical documentation enters the context window, the inferred language drifts to English. Context compaction (truncating conversation history) can delete language preferences entirely. AXOLENT's immutable context is immune to all three failure modes.

**Code reference:** `bridge/application/language/context.py`

### 2.5 Architecture Isolation

**Concept:** The Language Control Plane is a self-contained package with strict import boundaries enforced by automated tests running on every CI build. Detection libraries are encapsulated behind a Protocol interface and cannot leak into other application modules.

**Technical implementation:**
- All language-enforcement logic lives in `bridge/application/language/` (verifier, repair, stream guard, context, contract, enforcement facade, model profiles, backends)
- `LanguageDetectorBackend` Protocol: detection libraries (langdetect, Lingua, fast-langdetect) are implementation details confined to `backends.py`. The verifier and stream guard speak only to the Protocol, never to concrete libraries.
- Architecture test (`tests/test_architecture/test_langdetect_isolation.py`): scans every `.py` file in the codebase and fails the CI build if `import langdetect` or `from langdetect` appears anywhere except `backends.py`
- Second architecture test ensures domain-level language detection (`domain/language.py`, calibrated for short user inputs via marker-word heuristics) is never imported by the output verification layer (calibrated for long LLM outputs via n-gram profiles). Two different detection strategies, two different calibration targets, architecturally separated.
- `LanguageEnforcement` facade provides a single entry point for consumers, preventing verification/repair logic from scattering across the codebase
- `AuditLogPort` Protocol decouples the enforcement facade from infrastructure-layer audit logging. The `LanguageEnforcement` class declares a dependency on this Protocol (not on the concrete `write_audit_log` function from infrastructure). The concrete adapter is injected via constructor in `main.py`. This satisfies hexagonal architecture: the application layer defines the port, the infrastructure layer provides the adapter, and neither imports the other's internals.

**Differentiation from competition:** No other major AI assistant separates language handling into a dedicated subsystem with enforced import boundaries. Language logic in competitors (where it exists at all) is embedded in prompt templates or distributed across inference pipelines without isolation guarantees.

**Code reference:** `bridge/application/language/`, `bridge/application/language/enforcement.py` (AuditLogPort Protocol), `bridge/tests/test_architecture/test_langdetect_isolation.py`

---

## 3. Trinity Memory with Auto-Migration

AXOLENT implements a three-layer memory architecture modeled on cognitive science's established taxonomy of human memory systems (Tulving, 1972; Anderson, 1983). Each layer is a dedicated domain entity with its own data model, lifecycle, and query semantics.

### 3.1 Three Memory Layers

**Episodic Memory** stores concrete events with timestamps and context. Each entry records what happened, when, and in what setting. Example: "User asked about acquirers on 2026-05-07." Entries carry a user-assigned or auto-scored importance rating (1-10) and an optional context dict for workspace tags or source attribution.

**Semantic Memory** stores generalized facts extracted from episodic experience. These are user preferences, learned patterns, and factual knowledge that has been abstracted from specific events. Example: "User prefers short answers." Entries include a category field (fact, person, preference, project) for structured retrieval.

**Procedural Memory** stores repeatable action patterns the system has learned. These are skills with named identifiers and usage counters. Example: "When user asks for code, always respond with a code block." Entries track `skill_name` and `usage_count` for lifecycle management.

**Technical implementation:**
- All three layers are `frozen=True, slots=True` dataclasses in `bridge/domain/memory/`
- Each layer has a unique ID prefix for automatic layer detection: `ep_` (episodic), `sem_` (semantic), `pro_` (procedural)
- Serialization/deserialization via `to_dict()` / `from_dict()` class methods for JSONL and SQLite persistence
- Timestamps are ISO-8601 UTC strings throughout
- The `MemoryService` coordinator in `bridge/application/memory_service.py` provides the user-facing API (remember, recall, list, forget, get) and delegates to the storage backend. It detects the memory layer automatically from the ID prefix, so callers never need to specify which layer an entry belongs to.

### 3.2 Cross-Layer Migration

The three layers are not isolated silos. The system implements defined migration paths between layers:

**Episodic to Semantic:** When repeated episodic events share a common pattern (e.g., the user corrects the same output style three times), the pattern is abstracted into a semantic fact. The original episodes remain intact as provenance.

**Semantic to Procedural:** When a semantic fact reaches sufficient confidence and is confirmed by the user, it graduates to a procedural skill that the system can apply automatically. The skill tracks its usage count and can be versioned.

**Procedural back to Semantic/Episodic:** When a procedural skill encounters contradictions (user corrections that conflict with the skill), it can be demoted back to semantic for re-evaluation, or new episodic entries are created to capture the changed behavior.

These migration paths are the foundation for Skill-Compression (Section 5). The memory layers provide the storage substrate; Skill-Compression provides the pattern recognition and hypothesis management that drives migration decisions.

### 3.3 Differentiation from Competition

As of May 2026, no major AI assistant consumer product implements a three-layer memory taxonomy with defined cross-layer migration:

- **ChatGPT Memory** (OpenAI): single flat memory layer. Facts are stored as plain-text entries. No distinction between episodic events, semantic facts, and procedural skills. No migration lifecycle.
- **Claude Memory** (Anthropic): project-scoped context files (CLAUDE.md) plus user-level memory entries. Two scopes but one layer type. No episodic/procedural distinction.
- **Gemini Memory** (Google): conversation-derived preferences. Single layer, no taxonomy.
- **MemGPT/Letta** (UC Berkeley, now Letta Inc.): OS-inspired core/archival/recall model. This is the closest prior art, but the distinction is between context window management layers (what fits in the prompt vs. what is paged out), not between cognitive memory types. MemGPT does not implement procedural memory with usage-tracked skills, and does not define migration rules between memory types.
- **Mem0, Zep, LangMem:** Framework-level memory layers that support episodic/semantic/procedural as categories. These are infrastructure tools for developers, not end-user AI assistant products. They provide storage and retrieval primitives but do not implement the migration logic, hypothesis lifecycle, or skill-compression pipeline that AXOLENT builds on top of the three-layer substrate.

The academic literature (Tulving, Anderson, the "Memory in the Age of AI Agents" survey) establishes the three-layer taxonomy as theoretically sound. Several research systems (Generative Agents by Park et al., 2023; MemRL, 2026; MemEvolve, 2025) implement variations. AXOLENT's contribution is the integration of this taxonomy into a production local-first AI assistant with defined migration rules, frozen immutable domain entities, and a hypothesis-based lifecycle for cross-layer promotion.

**Code reference:** `bridge/domain/memory/episodic.py`, `bridge/domain/memory/semantic.py`, `bridge/domain/memory/procedural.py`, `bridge/application/memory_service.py`

---

## 4. Auto-Routing with Dynamic Model Selection

AXOLENT routes each user request to the most appropriate LLM model and provider combination. Unlike single-provider assistants that are locked to one model family, AXOLENT treats model selection as a per-request decision based on task classification, user constraints, and model capability profiles.

### 4.1 Task Classification (6 Slots)

The `TaskRouter` classifies each user message into one of six task slots using a three-stage deterministic heuristic (no LLM call for classification):

1. **Explicit markers:** Prefix commands (`/code`, `/reason`, `/research`, etc.) override all heuristics.
2. **Pattern and keyword matching:** YAML-configured patterns and keywords are scored per slot. Code blocks (`\`\`\``) receive a heavy score boost (100 points). Keywords receive 1 point each, patterns 3 points each. Minimum keyword thresholds prevent spurious matches. Word count filters constrain slots to appropriate input lengths.
3. **Fallback:** Messages that match no slot default to CHAT.

The six slots are: **CHAT** (general conversation), **CODE** (programming tasks), **REASON** (analytical/reasoning tasks), **CREATIVE** (creative writing), **QUICK** (short factual queries), and **RESEARCH** (deep research tasks).

On score ties, a fixed priority order applies: CODE > REASON > RESEARCH > CREATIVE > QUICK > CHAT. This is deterministic and auditable.

**German input normalization:** The classifier normalizes ASCII-encoded German umlauts (ae/oe/ue) to proper Unicode umlauts before keyword matching. This ensures that users typing on ASCII-only keyboards trigger the same slots as users with German keyboard layouts. The normalization uses a safe whitelist approach: `ss` to `eszett` conversion applies only to an explicit list of unambiguous German stems (e.g., "Strasse" but not "processing").

### 4.2 Model Resolution (Priority Cascade)

Once a task slot is determined, the router resolves which model to use via a priority cascade:

1. **User override per slot:** The user can set a preferred model for a specific slot (e.g., "use Opus for CODE tasks"). Stored in SQLite via `ModelService`.
2. **User override global:** A single model preference that applies to all slots.
3. **Slot default:** YAML-configured default model per slot (e.g., CODE defaults to a high-capability model, QUICK defaults to a fast model).
4. **System default:** Fallback when all other sources are empty.

All returned model IDs are canonical (e.g., `claude-opus-4-7`), never aliases. This prevents pool-key duplication when the same model is referenced by different names.

### 4.3 Multi-Provider Routing

The `ProviderRouter` maintains a registry of all registered LLM providers and routes each request to the appropriate one. Registered providers include Claude (Anthropic), OpenAI, Gemini (Google), Mistral, Groq, and Ollama (local models).

**Model compatibility guard:** When a model ID is passed that belongs to a different provider family (e.g., user set `/setmodel opus` globally but the current request targets Ollama), the router silently replaces the model with `None` so the target provider uses its own default. This prevents HTTP 404 errors from providers that do not recognize foreign model namespaces. Compatibility is checked via `ModelRegistry` metadata that maps each model to its provider family.

**Provider capabilities:** Each provider exposes a `ProviderCapabilities` object describing what it supports (streaming, function calling, vision, etc.). This metadata is available at runtime for routing decisions.

### 4.4 User Constraint Integration

User-level constraints influence routing decisions:

- **Privacy (local-first):** Users who require local processing are routed to Ollama or other local providers. No data leaves the machine.
- **Cost awareness:** Users can prefer cheaper models for routine tasks while reserving expensive models for complex reasoning.
- **Provider preference:** Users can exclude specific providers entirely (e.g., for policy reasons).

These constraints are not merely configuration; they are enforced at the routing layer before any request reaches a provider.

### 4.5 Differentiation from Competition

As of May 2026, the LLM routing landscape includes several infrastructure-level routing products (OpenRouter, LiteLLM, Portkey, Bifrost, Martian). These are API gateways for developers: they unify provider APIs and offer cost/latency optimization across providers. They are not end-user AI assistants.

AXOLENT's routing differs in three ways:

1. **Task-aware classification before routing:** The task slot determines which model capability profile is needed. A CODE task routes to a different model than a QUICK task, even from the same user in the same session. Gateway routers like OpenRouter and LiteLLM do not classify tasks; they route based on explicit model selection or simple cost rules.

2. **Integrated into a local-first assistant:** The routing decision is made locally, respects local-only constraints, and operates within the Mode B architecture (no cloud relay). API gateways are cloud services by definition.

3. **Per-user model memory:** User model preferences are stored locally in SQLite and persist across sessions. The routing cascade (slot override > global override > slot default > system default) adapts to each user's needs without requiring configuration files. Gateway routers have no concept of per-user preference persistence.

No major consumer AI assistant (ChatGPT, Claude, Gemini, Copilot, Perplexity) offers transparent multi-provider routing. Each is locked to its own model family. AXOLENT is the first local-first AI assistant to implement deterministic task classification with multi-provider model resolution.

**Code reference:** `bridge/domain/task_slot.py` (6 slots), `bridge/application/task_router.py` (classification + model resolution), `bridge/application/provider_router.py` (multi-provider routing + compatibility guard), `bridge/config/task_slots.yaml` (slot configuration)

---

## 5. Skill-Compression as Hypothesis System

Skill-Compression is AXOLENT's pattern-recognition engine. It observes user interactions over time, detects recurring behavioral patterns, models them as falsifiable hypotheses with structured evidence, and (after user confirmation) applies them as automated skills. The system follows the principle: algorithms find candidates, evidence decides, user approves.

### 5.1 Architecture (7 Layers)

The Skill-Compression engine is organized in seven layers, each with a defined responsibility boundary:

**Layer 1: Event Normalizer.** Reads each user-bot interaction and extracts structured fields: intent, domain, format type, constraints (duration, length, funnel stage, audience), scope (project, client), language, correction keywords, and re-formulation signals. Classification is entirely rule-based (regex patterns, keyword matching), with no LLM call and no embedding. Each event receives a deterministic SHA-256 fingerprint hash computed over the canonical JSON representation of its structured fields.

**Layer 2: Algorithmic Candidate Layer.** Four independent pattern-detection algorithms produce candidate patterns. None of them is a truth source; all are proposal generators that feed into the Evidence Ledger.
- *Fingerprint Similarity:* Compares events via weighted structured-field similarity (intent 30%, domain 20%, constraints 20%, format 15%, scope 15%). Language is a hard filter, not a weighted field. Threshold for candidacy: > 0.7. Field comparison uses type-appropriate strategies: prefix matching for hierarchical intents, exact matching for domains, dict overlap for constraints, scope-specific similarity for project/client contexts.
- *N-Gram Sliding Window:* Slides windows of size 3, 4, and 5 over chronological event sequences. Extracts recurring subsequences as workflow candidates (e.g., "create_ad, review, publish"). Patterns are identified by SHA-256 hash and counted by frequency. Only patterns with 2+ occurrences are retained.
- *Markov Chain:* Per-user first-order transition matrix between action types (domain.intent states). Fully incremental: each new event updates the matrix without recomputation. Predicts most likely next actions for proactive skill matching. Privacy-friendly: stores only transition counts, not message content.
- *Elo Rating:* Pattern confidence is computed using the chess Elo system. Initial rating: 1500. Each pattern application is a "match" against a request with its own difficulty rating. Successful application (user does not correct) raises the pattern's rating; failure (user corrects) lowers it. The magnitude of change depends on the difficulty differential: losing against an easy request causes a larger rating drop than losing against a hard request. This emerges naturally from the Elo formula. Request difficulty ratings are tracked per fingerprint hash and update bidirectionally with each match outcome.

**Layer 3: Evidence Ledger.** Structured proof records per hypothesis. Each evidence entry links to a hypothesis version, an episode, a request, and a response. Signal types include: no_correction (positive), bookmark (positive), explicit_confirm (positive), correction (negative), rejection (negative), learn_command (strongly positive). Signal strength is a float, not binary.

**Layer 4: Pattern Judge.** Manages the hypothesis lifecycle (7 statuses, see 5.2) and enforces promotion/demotion rules. Implements Skill Collision Detection: when two skills with overlapping scope both match a request, the more specific scope wins automatically. When scopes are equally specific, the system asks the user rather than deciding autonomously. Negative patterns receive 2x weight in contradiction scoring, scope-differentiated.

**Layer 5: SkillMatcher.** Applies confirmed or active hypotheses to new requests. "Ask Before Applying" is the default mode. Auto-Apply activates only after scope-differentiated Elo thresholds are met (see 5.3). Every skill application produces a visible indicator to the user.

**Layer 6: UI / Explainer.** User-facing skill profile ("Dein Profil") with bullet-point display, version history, and action buttons. 8 explainer question types for transparency (see 5.5). 4 chat shortcuts: `/skills`, `/skill X`, `/forget X`, `/learn`.

**Layer 7: Privacy Safeguards.** Hard red lines enforced in code, not just documented (see 5.6).

### 5.2 Hypothesis Lifecycle

Each detected pattern is modeled as a `Hypothesis`: a `frozen=True, slots=True` dataclass with full lifecycle state. The type field is TEXT, not an enum, to allow future extension without schema migration.

The lifecycle has 7 statuses:

1. **candidate** (1-2 evidence items, internal only, user does not see)
2. **suggested** (3-5 evidence items across 2+ sessions, system asks user for confirmation)
3. **confirmed** (user confirmed, applied with "Ask Before Applying")
4. **active** (auto-apply threshold reached, applied without prompting)
5. **needs_review** (contradiction detected, returns to user-question mode)
6. **paused** (user manually paused, remains stored)
7. **archived** (180+ days without application, FSRS decay threshold reached)

Plus **retired** (user invoked `/forget`, tombstone created for 30 days or permanently).

**Skill versioning:** Skills have version numbers with `predecessor_context`. When a skill evolves (e.g., from "30s retargeting" to "45s brand awareness" after repeated corrections), a new version is created. Historical evidence remains linked to the old version. The new version references old evidence as predecessor context, not as direct support. Users can inspect the version history and revert.

### 5.3 Trigger and Promotion Rules

Three conditions must be met simultaneously before a pattern becomes a candidate:
1. Structured fingerprint similarity > 0.7 across repetitions
2. At least 3 confirmations across at least 2 sessions
3. Generation time > 10 seconds OR output volume > 500 tokens

Auto-Apply thresholds are scope-differentiated and risk-aware:

| Pattern Type | Scope Breadth | Min. Confirmations | Min. Elo Rating |
|---|---|---|---|
| Negative | Specific (e.g., "client emails") | 2 | 1650 |
| Negative | Domain (e.g., "business texts") | 4 | 1700 |
| Negative | Global (always, everywhere) | 6 | 1750 |
| Preference | Any | 5 | 1700 |
| Procedural | Any | 8 | 1800 |

Additional conditions for Auto-Apply: confirmations span at least 2 distinct sessions; no active tombstone; skill is not in status paused or needs_review.

### 5.4 Decay and Tombstones

**Decay (FSRS v7):** Each skill has individual forgetting-curve parameters learned via the Free Spaced Repetition Scheduler (FSRS v7). The scheduler tracks each skill's usage rhythm, success/failure history, and time since last application. Seasonal patterns (regular but infrequent use, e.g., monthly reporting) are recognized and exempt from aggressive decay. Absolute floor: 180 days without use triggers archive status.

**User-created skills are decay-immune.** Skills created via `/learn`, manual UI entry, or explicit user confirmation on a system suggestion never auto-decay. Only the user can remove them via `/forget`.

**Tombstones:** When a user invokes `/forget`, a tombstone record is created. Default duration: 30 days. The user can choose "never again" for a permanent tombstone. During the tombstone period, the system will not re-learn the same pattern. Tombstone matching uses pattern hash, scope hash, and similarity threshold (> 0.85 = treated as same pattern).

**Skill library cap:** Maximum 50 active skills per user. When the cap is reached, the next candidate triggers a user dialog with three options: add anyway (raises personal cap), replace an existing skill (sorted by confidence ascending), or reject the candidate. The user decides; the system never silently evicts skills.

### 5.5 Explainer Transparency

The system provides 8 explainer question types, accessible via `/explain X` or the "Why?" UI button:

1. "What pattern was detected?" (observation)
2. "Why was this NOT promoted to a skill?" (negative-decision reasoning)
3. "Why was this promoted to a skill?" (positive-decision reasoning)
4. "When was drift detected?" (evidence timeline)
5. "What would be needed for this pattern to become trustworthy again?" (structural measure)
6. "What lessons has the system learned from pattern X?" (lessons learned)
7. "Where does this skill NOT apply?" (scope boundaries)
8. "What evidence speaks AGAINST this skill?" (counter-evidence)

### 5.6 Privacy Safeguards

**No-Model-Secret Rule (multi-layered):** The system never stores in hypotheses: prices (regex: currency symbols + numbers), API tokens (regex: typical token patterns like sk-, ghp_), passwords (heuristic: proximity to password keywords), private identifiers (email addresses, phone numbers, IBANs), raw data or internal prompt fragments, personal data of third parties. Implementation is three-tiered: (1) allowlist of permitted skill fields, (2) secret scanner with regex patterns, (3) heuristic filter for unrecognized long number strings. On doubt: do not store.

**Healthcare filter:** No behavioral-clinical phenotyping. The system does not infer health conditions from writing patterns, emotional states from interaction frequency, or mental health status from any signal. This is a hard red line enforced in code.

**Tombstones with permanent option:** Deleted patterns remain blocked for 30 days by default, permanently if the user chooses.

**Nudge self-commitment:** A comprehensive negative list of manipulative patterns that the system will never apply, including: political personalization, FOMO patterns, artificial urgency, relationship suggestions, engagement loops, confirmshaming, social pressure, behavioral inferences not serving user goals, and data export to third parties.

**Encryption at rest:** The hypothesis database uses SQLCipher (AES-256) with key management via the OS keychain (macOS Keychain, Windows Credential Manager, Linux Secret Service).

### 5.7 Storage Schema

The Skill-Compression engine uses 7 dedicated SQLite tables: `hypotheses` (core records with Elo/FSRS/Bayesian state), `hypothesis_aliases` (dynamic term pool), `hypothesis_evidence` (structured proof ledger), `hypothesis_versions` (version history with predecessor context), `hypothesis_tombstones` (deleted patterns with TTL), `hypothesis_local_eval_set` (smoke-test example pairs per hypothesis), and `pattern_difficulty` (Elo rating per fingerprint for difficulty-aware updates).

The type column in `hypotheses` is TEXT, not an enum or CHECK constraint. This permits future pattern type extensions (context, style, outcome) without schema migration.

### 5.8 Differentiation from Competition

Several research systems implement parts of the Skill-Compression concept:

- **Voyager** (Wang et al., 2023): Skill library pattern for Minecraft agents. Skills are executable code functions stored and retrieved by description similarity. No hypothesis lifecycle, no evidence ledger, no user confirmation, no scope tracking, no decay model.
- **Generative Agents** (Park et al., 2023): Memory-based behavioral agents with reflection. Reflection produces higher-level insights from observations. No falsifiable hypothesis model, no Elo confidence, no user-facing skill management.
- **MemGPT/Letta**: OS-inspired memory paging. No pattern detection, no hypothesis lifecycle, no skill application.
- **Reflexion** (Shinn et al., 2023): Self-reflection for task improvement. Stores verbal feedback as memory. No structured evidence ledger, no cross-session pattern detection, no user-controlled lifecycle.

No known system (academic or production) combines all of the following in a single architecture:

- Local-only processing (no server, no cloud)
- Falsifiable hypothesis model with 7-status lifecycle
- Structured evidence ledger with per-version provenance
- Scope-differentiated auto-apply thresholds
- FSRS v7 for individual forgetting curves per skill
- Elo rating for pattern confidence with request-difficulty coupling
- User-confirmation lifecycle with Ask-Before-Applying default
- Pattern tombstones with 30-day default and permanent option
- 8-type explainer for full transparency
- Healthcare filter as hard red line
- Multi-layered secret scanner (allowlist + regex + heuristic)
- Encryption at rest via SQLCipher

This combination has been independently confirmed as novel in external code review.

**Code reference:** `bridge/application/skill_compression/` (complete subsystem), `bridge/application/skill_compression/event_normalizer.py` (Layer 1), `bridge/application/skill_compression/fingerprint_matcher.py` (Layer 2: fingerprint similarity), `bridge/application/skill_compression/ngram_extractor.py` (Layer 2: N-gram), `bridge/application/skill_compression/markov_chain.py` (Layer 2: Markov), `bridge/application/skill_compression/elo_rating.py` (Layer 2: Elo), `bridge/application/skill_compression/hypothesis_storage.py` (Layer 3-4: schema + CRUD)

---

## 6. Architecture Guards

Architectural invariants across all six axes are enforced by automated structural tests that run on every commit. Guards are not conventions; they are CI-enforced constraints that cause immediate build failure on violation.

### 6.1 Detection Library Isolation Test

**Test:** `test_langdetect_only_imported_in_backends_module`

Scans every Python file in the repository (excluding `.venv`, `__pycache__`, `.pytest_cache`) for direct imports of the `langdetect` library. The only permitted location is `application/language/backends.py`. Any import elsewhere causes immediate CI failure.

**Rationale:** In a prior iteration, a distributed `_CHAR_HINTS` state pattern (where langdetect initialization leaked across modules) caused multi-day debugging sessions. The architecture test ensures this class of bug is structurally impossible.

### 6.2 Domain/Application Boundary Test

**Test:** `test_domain_language_not_used_in_output_verifier`

Ensures that `domain.language` (the short-input marker-word detector) is never imported by the output verifier or stream guard. These components require n-gram-profile-based detection suited for long outputs. Using the wrong detector would produce unreliable results on LLM outputs.

### 6.3 Hexagonal Import Contracts

Beyond the language-specific tests, the project uses `import-linter` with three contracts:
1. Hexagonal layer ordering (presentation > application > infrastructure > domain)
2. Domain purity (no I/O imports)
3. Presentation isolation (no direct infrastructure imports)

These ensure the Language Control Plane (living in `application/`) cannot accidentally import presentation-layer or leak infrastructure details upward.

### 6.4 Skill-Compression Guards

The Skill-Compression engine requires its own set of architecture guards:

| Guard | What it prevents |
|---|---|
| `test_hypothesis_is_frozen_dataclass` | Mutations of Hypothesis objects |
| `test_no_secret_patterns_in_hypotheses` | Token/price/IBAN leaks in skill storage |
| `test_decay_immune_skills_not_aged` | Auto-decay applied to user-created skills |
| `test_tombstone_blocks_relearning` | Spontaneous re-learning despite active tombstone |
| `test_max_active_skills_enforced` | Skill library exceeding 50 without user confirmation |
| `test_no_phenotyping_inferences` | Healthcare patterns being materialized |
| `test_sqlcipher_enabled_in_prod` | Database opened without encryption |
| `test_pattern_type_is_text_not_enum` | CHECK constraint on type column (prevents future extension) |

---

## 7. Prior Art Claim

This document establishes the following prior art as of 2026-05-20 (updated 2026-05-21):

**Language Control Plane (Section 2):**
1. The concept of a dedicated, architecturally isolated Language Control Plane in an AI assistant
2. Post-hoc output language verification as a system-level guarantee (not model-level hope)
3. Automatic repair-loop re-query on language verification failure with hard-capped attempts
4. Mid-stream language drift detection with self-calibrating confidence thresholds
5. Immutable language context as a frozen dataclass that prevents mid-pipeline mutation
6. `AuditLogPort` Protocol for hexagonal decoupling of enforcement from infrastructure audit logging
7. `MappingProxyType` for container-level immutability of detection distributions
8. `had_dissent` field for dissent-tracking in multi-backend detection orchestration
9. `StreamGuard` self-calibration wired into the production streaming path

**Trinity Memory (Section 3):**
10. Three-layer memory taxonomy (episodic/semantic/procedural) with frozen domain entities and automatic layer detection via ID prefix in a local-first AI assistant
11. Defined cross-layer migration rules between episodic, semantic, and procedural memory with provenance preservation

**Auto-Routing (Section 4):**
12. Deterministic 6-slot task classification with three-stage heuristic (explicit markers, pattern/keyword scoring, fallback) for per-request model selection in a local-first AI assistant
13. Multi-provider model routing with model-family compatibility guard and per-user preference persistence
14. German input normalization with safe whitelist approach for ASCII-to-umlaut conversion in task classification

**Skill-Compression (Section 5):**
15. Falsifiable hypothesis model with 7-status lifecycle for user-behavior pattern recognition in a local-first AI assistant
16. Four-algorithm candidate layer (fingerprint similarity, N-gram sliding window, Markov chain, Elo rating) as proposal generators feeding a structured evidence ledger
17. Scope-differentiated auto-apply thresholds with risk-aware Elo rating minimums
18. FSRS v7 for individual per-skill forgetting curves with seasonal pattern recognition
19. Pattern tombstones with 30-day default and permanent option, matched via fingerprint and scope hashes
20. 8-type explainer system for hypothesis transparency
21. Multi-layered secret scanner (allowlist + regex + heuristic) preventing sensitive data in skill storage
22. Healthcare filter as hard-coded red line preventing behavioral-clinical phenotyping

These concepts are implemented in working code, covered by 1900+ tests, and released under AGPL-3.0.

---

## 8. References

| Reference | Location | Access |
|---|---|---|
| Language Control Plane package | `bridge/application/language/` | Public (AGPL-3.0) |
| Architecture isolation test | `bridge/tests/test_architecture/test_langdetect_isolation.py` | Public (AGPL-3.0) |
| Trinity Memory domain entities | `bridge/domain/memory/` | Public (AGPL-3.0) |
| Memory Service coordinator | `bridge/application/memory_service.py` | Public (AGPL-3.0) |
| Task classification (6 slots) | `bridge/domain/task_slot.py`, `bridge/application/task_router.py` | Public (AGPL-3.0) |
| Multi-provider routing | `bridge/application/provider_router.py` | Public (AGPL-3.0) |
| Slot configuration | `bridge/config/task_slots.yaml` | Public (AGPL-3.0) |
| Skill-Compression engine | `bridge/application/skill_compression/` | Public (AGPL-3.0) |
| Hypothesis storage schema | `bridge/application/skill_compression/hypothesis_storage.py` | Public (AGPL-3.0) |
| Architecture documentation | `docs/ARCHITECTURE.md` | Public (AGPL-3.0) |
| Public/Private boundary | `docs/PUBLIC_PRIVATE_BOUNDARY.md` | Public (AGPL-3.0) |
| Language competition analysis | `docs/COMPETITION_LANGUAGE.md` | Public (AGPL-3.0) |
| Detailed competitor comparison | `cosmo-ai-assistant-language-comparison-2026-05-20.md` | Internal (Obsidian Vault) |
| Skill-Compression specification | `spec-skill-compression-final-2026-05-20.md` | Internal (Obsidian Vault) |
| Codex LCP review report | `codex-lcp-review-report-2026-05-20.md` | Internal (Obsidian Vault) |
| ADR: Language resolution contract | `docs/adr/0004-language-resolution-contract.md` | Public (AGPL-3.0) |
| ADR: Execution Kernel | `docs/adr/0003-execution-kernel-architecture.md` | Public (AGPL-3.0) |

---

## 9. Scope and Limitations

This defensive publication covers architectural concepts and their implementation across six axes: Mode B local-first architecture, Language Control Plane, Trinity Memory, Auto-Routing, Skill-Compression, and Architecture Guards.

It does not make performance claims, accuracy guarantees, or marketing promises. The system is under active development. Specific thresholds, detection backends, model routing weights, and skill-promotion strategies may change. The architectural patterns (verify-repair-guard-isolate for language; episodic-semantic-procedural for memory; classify-route-constrain for model selection; observe-hypothesize-evidence-apply for skill compression) are the stable claims.

This publication is made under AGPL-3.0. The code is open source and freely inspectable.
