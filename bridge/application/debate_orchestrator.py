"""Debate orchestrator: multi-AI debate feature (R10).

Queries multiple providers in parallel with the same question and collects responses.
Crash-resilient: a crashing provider does not stop the others.
Consensus/dissent analysis via heuristic + LLM-as-Judge final review.

Provider deduplication (since R10 fix):
When multiple providers use the same backend model (e.g. claude_persistent
and claude both use the Claude CLI), only one per group is used in the debate.
This prevents skewed consensus analyses and token waste.

Final review layer (since R10 extension):
After the parallel responses, an LLM-as-Judge call evaluates all
responses and delivers a core takeaway with pros/cons.
Judge provider: claude_persistent (fallback: ollama_local with quality warning).
Bias mitigation: provider names are anonymized in the judge prompt.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from application.provider_router import ProviderRouter

log = logging.getLogger(__name__)

# Configuration via environment
DEBATE_TIMEOUT_SECONDS: int = 60
_DEBATE_PROVIDERS_RAW: str = os.getenv("DEBATE_PROVIDERS", "")

# Sentinel chat ID for judge calls: separate conversation context
# so the judge does not have the debate response in its context.
_JUDGE_CHAT_ID_OFFSET: int = 900_000_000

# Provider groups: providers that use the same backend model.
# Per group only the first available provider is used in the debate.
# Order = priority (first entry preferred).
PROVIDER_GROUPS: dict[str, list[str]] = {
    "claude": ["claude_persistent", "claude"],  # both use Claude CLI
}

# Reverse lookup: provider_name -> group_name (or None if standalone)
_PROVIDER_TO_GROUP: dict[str, str] = {}
for _group_name, _members in PROVIDER_GROUPS.items():
    for _member in _members:
        _PROVIDER_TO_GROUP[_member] = _group_name


def _get_configured_providers() -> list[str] | None:
    """Parse DEBATE_PROVIDERS env var. None = use all available."""
    if not _DEBATE_PROVIDERS_RAW.strip():
        return None
    return [p.strip() for p in _DEBATE_PROVIDERS_RAW.split(",") if p.strip()]


def deduplicate_providers(available: list[str]) -> list[str]:
    """Deduplicate providers that use the same backend model.

    Per PROVIDER_GROUPS group, only the first available provider is kept.
    Standalone providers (not in a group) are always kept.

    Args:
        available: List of available provider names.

    Returns:
        Deduplicated list (order preserved).
    """
    selected: list[str] = []
    used_groups: set[str] = set()

    for provider in available:
        group = _PROVIDER_TO_GROUP.get(provider)
        if group:
            if group not in used_groups:
                selected.append(provider)
                used_groups.add(group)
                log.debug("Provider dedup: %s represents group '%s'", provider, group)
            else:
                log.debug(
                    "Provider dedup: %s skipped (group '%s' already represented)",
                    provider,
                    group,
                )
        else:
            # Standalone provider: always keep
            selected.append(provider)

    if len(selected) < len(available):
        log.info(
            "Provider dedup: %d -> %d providers (%s)",
            len(available),
            len(selected),
            selected,
        )

    return selected


@dataclass(frozen=True)
class ProviderEvaluation:
    """Evaluation of a single provider response by the judge.

    Attributes:
        provider: Provider name (real name, after de-anonymization).
        pros: List of positive aspects of the response.
        cons: List of negative aspects of the response.
    """

    provider: str
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FinalVerdict:
    """Result of the LLM-as-Judge final review.

    Attributes:
        winner: Provider name of the winner (or "tie" on a draw).
        recommendation: Core takeaway covering all aspects of the question.
        synthesis: Content synthesis combining the best of all responses.
        evaluations: Per-provider pros/cons evaluation.
        reasoning: 1-2 sentences on why this winner was chosen.
        judge_provider: Which provider made the judge call.
        judge_quality_warning: Warning if a weaker judge was used.
    """

    winner: str
    recommendation: str
    synthesis: str = ""
    evaluations: list[ProviderEvaluation] = field(default_factory=list)
    reasoning: str = ""
    judge_provider: str = ""
    judge_quality_warning: str | None = None


@dataclass(frozen=True)
class DebateResult:
    """Result of a multi-AI debate.

    Attributes:
        question: The question asked.
        responses: Provider name -> response text (successful providers).
        errors: Provider name -> error message (crashed providers).
        consensus_analysis: Consensus/dissent analysis (optional).
        final_verdict: LLM-as-Judge evaluation (optional, None if judge fails).
        duration_seconds: Total duration of the debate.
        providers_queried: List of all queried providers.
    """

    question: str
    responses: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    consensus_analysis: Optional[str] = None
    final_verdict: Optional[FinalVerdict] = None
    duration_seconds: float = 0.0
    providers_queried: list[str] = field(default_factory=list)


class DebateOrchestrator:
    """Orchestrates multi-AI debates across multiple providers.

    Queries all available (or configured) providers in parallel,
    collects responses with timeout protection, and creates a consensus analysis.

    Args:
        provider_router: The ProviderRouter with registered providers.
        timeout_seconds: Max wait time per provider (default: 60s).
    """

    def __init__(
        self,
        provider_router: ProviderRouter,
        timeout_seconds: int = DEBATE_TIMEOUT_SECONDS,
    ) -> None:
        self.provider_router = provider_router
        self.timeout_seconds = timeout_seconds

    def _select_providers(self) -> list[str]:
        """Determine which providers to use for the debate.

        Priority:
        1. DEBATE_PROVIDERS env var (if set)
        2. All available providers

        In both cases, deduplication follows: providers using the same
        backend model are reduced to one per group.

        Returns:
            Deduplicated list of provider names.
        """
        configured = _get_configured_providers()
        if configured is not None:
            available = set(self.provider_router.list_available())
            selected = [p for p in configured if p in available]
            if not selected:
                log.warning(
                    "No configured DEBATE_PROVIDERS available: %s. Available: %s",
                    configured,
                    list(available),
                )
            return deduplicate_providers(selected)

        all_available = self.provider_router.list_available()
        return deduplicate_providers(all_available)

    async def _query_provider(
        self,
        provider_name: str,
        question: str,
        user_id: int,
        chat_id: int,
    ) -> tuple[str, str | None, str | None]:
        """Query a single provider with timeout.

        Returns:
            Tuple: (provider_name, response_text_or_None, error_or_None)
        """
        try:
            response = await asyncio.wait_for(
                self.provider_router.route(
                    prompt=question,
                    system_prompt=(
                        "Answer concisely and informatively. "
                        "Keep it to 2-4 sentences if possible."
                    ),
                    provider_name=provider_name,
                    timeout_seconds=self.timeout_seconds,
                    user_id=user_id,
                    chat_id=chat_id,
                ),
                timeout=self.timeout_seconds + 5,
            )
            if response.success:
                return (provider_name, response.text, None)
            else:
                return (provider_name, None, response.error or "Unknown error")
        except asyncio.TimeoutError:
            return (provider_name, None, f"Timeout after {self.timeout_seconds}s")
        except Exception as exc:
            return (provider_name, None, str(exc))

    def _analyze_consensus(self, responses: dict[str, str]) -> str:
        """Simple consensus heuristic (Phase 1, no LLM judge).

        Compares response lengths and simple word overlap analysis.

        Args:
            responses: Provider name -> response text.

        Returns:
            Brief consensus/dissent assessment.
        """
        if len(responses) < 2:
            return "Only one provider responded. No comparison possible."

        texts = list(responses.values())

        # Word sets for overlap analysis
        word_sets: list[set[str]] = []
        for text in texts:
            words = set(text.lower().split())
            significant = {w for w in words if len(w) > 3}
            word_sets.append(significant)

        # Pairwise overlap (Jaccard similarity)
        overlaps: list[float] = []
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                if not union:
                    overlaps.append(0.0)
                    continue
                intersection = word_sets[i] & word_sets[j]
                overlaps.append(len(intersection) / len(union))

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

        if avg_overlap > 0.35:
            return (
                f"The providers largely agree on content "
                f"(word overlap: {avg_overlap:.0%}). "
                f"High agreement on core statements."
            )
        elif avg_overlap > 0.20:
            return (
                f"The providers show partial agreement "
                f"(word overlap: {avg_overlap:.0%}). "
                f"Core statements similar, but different emphasis."
            )
        else:
            return (
                f"The providers give significantly different responses "
                f"(word overlap: {avg_overlap:.0%}). "
                f"Compare the responses above for different perspectives."
            )

    def _build_judge_prompt(
        self,
        question: str,
        responses: dict[str, str],
    ) -> tuple[str, dict[str, str]]:
        """Build the judge prompt with anonymized provider names.

        Bias mitigation: provider names are replaced with neutral labels.
        The judge only sees "Answer A", "Answer B", etc.

        Args:
            question: The original question.
            responses: Provider name -> response text.

        Returns:
            Tuple: (prompt_text, label_to_provider_mapping)
                label_to_provider_mapping: e.g. {"A": "claude_persistent", "B": "ollama_local"}
        """
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        label_to_provider: dict[str, str] = {}
        answer_blocks: list[str] = []

        for i, (provider_name, text) in enumerate(responses.items()):
            label = labels[i] if i < len(labels) else f"Z{i}"
            label_to_provider[label] = provider_name
            answer_blocks.append(f"--- Answer {label} ---\n{text.strip()}\n")

        answers_text = "\n".join(answer_blocks)

        prompt = (
            f"User question:\n{question}\n\n"
            f"The following answers were generated by different AI models.\n"
            f"Evaluate them neutrally and objectively.\n\n"
            f"{answers_text}\n"
            f"Your task:\n"
            f"1. Identify the strengths and weaknesses of each answer\n"
            f"2. Create a SYNTHESIS combining the best of all answers\n"
            f"3. The synthesis must be a standalone, complete answer "
            f"(not just 'A is better')\n"
            f"4. The key takeaway (key_takeaway) must cover ALL aspects of the question. "
            f"For multi-part questions (e.g. 'What is X and should I do Y?'), structure "
            f"the key takeaway so each sub-aspect is addressed.\n\n"
            f"IMPORTANT: Your ENTIRE response must be ONE SINGLE JSON object.\n"
            f"No text before, no text after, no markdown, no explanation.\n"
            f"Start directly with {{ and end with }}.\n\n"
            f"JSON schema (follow exactly):\n"
            f'{{"winner": "<letter of best answer or tie>", '
            f'"synthesis": "<Complete synthesized answer combining the best, '
            f'2-5 sentences, NEVER leave empty>", '
            f'"recommendation": "<Complete key takeaway covering all question aspects, '
            f'2-4 sentences>", '
            f'"evaluations": ['
            f'{{"label": "<letter>", "pros": ["..."], "cons": ["..."]}}, ...'
            f"], "
            f'"reasoning": "<1-2 sentences why this winner>"}}'
        )

        return prompt, label_to_provider

    @staticmethod
    def _extract_json_object(raw_text: str) -> str | None:
        """Extract a JSON object from arbitrary text.

        Strategies (in order):
        1. Entire text is valid JSON
        2. Remove markdown code block (```json ... ``` or ``` ... ```)
        3. Extract first { to last } (brace matching)

        Args:
            raw_text: Arbitrary text that may contain a JSON object.

        Returns:
            Extracted JSON string or None if no JSON found.
        """
        text = raw_text.strip()

        # Strategy 1: entire text is already valid JSON
        if text.startswith("{"):
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError:
                pass

        # Strategy 2: markdown code block (```json\n...\n``` or ```\n...\n```)
        codeblock_match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        if codeblock_match:
            candidate = codeblock_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Strategy 3: first { to matching } (brace counting)
        first_brace = text.find("{")
        if first_brace == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        for i in range(first_brace, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                if in_string:
                    escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        return None

        return None

    def _parse_judge_response(
        self,
        raw_text: str,
        label_to_provider: dict[str, str],
    ) -> FinalVerdict | None:
        """Parse the JSON response from the judge and map labels to provider names.

        Graceful: returns None on parse errors.
        Robust: extracts JSON even when the judge writes prose around it
        or uses a markdown code block.

        Args:
            raw_text: Raw response from the judge LLM.
            label_to_provider: Mapping label -> provider name.

        Returns:
            FinalVerdict or None if parsing fails.
        """
        log.debug("Judge raw response (%d chars): %s", len(raw_text), raw_text[:500])

        json_text = self._extract_json_object(raw_text)
        if json_text is None:
            log.warning(
                "Judge response: no JSON object extractable. First 300 chars: %s",
                raw_text[:300],
            )
            return None

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            log.warning("Judge response is not valid JSON: %s", json_text[:200])
            return None

        if not isinstance(data, dict):
            log.warning("Judge response is not a dict: %s", type(data))
            return None

        winner_label = data.get("winner", "")
        recommendation = data.get("recommendation", "")
        synthesis = data.get("synthesis", "")
        reasoning = data.get("reasoning", "")
        evaluations_raw = data.get("evaluations", [])

        # Winner label -> provider name
        if winner_label.lower() == "tie":
            winner = "tie"
        else:
            winner = label_to_provider.get(winner_label, winner_label)

        # Parse evaluations
        evaluations: list[ProviderEvaluation] = []
        for eval_item in evaluations_raw:
            if not isinstance(eval_item, dict):
                continue
            label = eval_item.get("label", "")
            provider_name = label_to_provider.get(label, label)
            pros = eval_item.get("pros", [])
            cons = eval_item.get("cons", [])
            if not isinstance(pros, list):
                pros = [str(pros)]
            if not isinstance(cons, list):
                cons = [str(cons)]
            evaluations.append(
                ProviderEvaluation(
                    provider=provider_name,
                    pros=pros,
                    cons=cons,
                )
            )

        return FinalVerdict(
            winner=winner,
            recommendation=str(recommendation),
            synthesis=str(synthesis),
            evaluations=evaluations,
            reasoning=str(reasoning),
        )

    async def final_review(
        self,
        question: str,
        responses: dict[str, str],
        user_id: int,
        chat_id: int,
    ) -> FinalVerdict | None:
        """Run the LLM-as-Judge final review.

        Strategy:
        1. Try claude_persistent as judge (highest quality)
        2. Fallback to ollama_local with quality warning
        3. On complete failure: None (caller falls back to heuristic)

        Bias mitigation: provider names are anonymized in the judge prompt.

        Args:
            question: The original question.
            responses: Provider name -> response text (at least 2 entries).
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.

        Returns:
            FinalVerdict or None if judge fails completely.
        """
        if len(responses) < 2:
            log.debug("Final review skipped: fewer than 2 responses")
            return None

        prompt, label_to_provider = self._build_judge_prompt(question, responses)

        judge_system_prompt = (
            "You are a neutral arbiter evaluating AI answers. "
            "You do not know the provider names and evaluate purely on quality: "
            "correctness, completeness, clarity, and relevance. "
            "ALWAYS respond with valid JSON, never with prose."
        )

        # Judge provider selection: claude_persistent > claude > ollama_local
        judge_candidates = ["claude_persistent", "claude", "ollama_local"]
        available = set(self.provider_router.list_available())

        # Exclude providers that participated in the debate to minimize
        # self-evaluation bias? No: with only 2 providers none would remain.
        # Instead: anonymization is sufficient as bias mitigation.

        judge_provider: str | None = None
        quality_warning: str | None = None

        for candidate in judge_candidates:
            if candidate in available:
                judge_provider = candidate
                break

        if judge_provider is None:
            log.warning("No judge provider available, final review skipped")
            return None

        if judge_provider == "ollama_local":
            quality_warning = "Local judge (Ollama), evaluation quality reduced"

        log.info("Final review: judge provider = %s", judge_provider)

        # Isolated conversation context for the judge:
        # offset on chat_id so the judge call does NOT go into the
        # user conversation (would create bias through previous debate response
        # and can disrupt JSON output because Claude responds in chat mode).
        judge_chat_id = chat_id + _JUDGE_CHAT_ID_OFFSET

        try:
            response = await asyncio.wait_for(
                self.provider_router.route(
                    prompt=prompt,
                    system_prompt=judge_system_prompt,
                    provider_name=judge_provider,
                    timeout_seconds=self.timeout_seconds,
                    user_id=user_id,
                    chat_id=judge_chat_id,
                ),
                timeout=self.timeout_seconds + 5,
            )

            if not response.success:
                log.warning(
                    "Judge call failed: %s (text=%r)",
                    response.error or "no text",
                    (response.text or "")[:200],
                )
                return None

            log.debug(
                "Judge response received (%d chars, %.1fs): %s",
                len(response.text),
                response.duration_seconds,
                response.text[:300],
            )

            verdict = self._parse_judge_response(response.text, label_to_provider)
            if verdict is None:
                log.warning(
                    "Judge response could not be parsed. Full response (%d chars): %s",
                    len(response.text),
                    response.text[:500],
                )
                return None

            # Add judge metadata (frozen dataclass, new object)
            return FinalVerdict(
                winner=verdict.winner,
                recommendation=verdict.recommendation,
                synthesis=verdict.synthesis,
                evaluations=verdict.evaluations,
                reasoning=verdict.reasoning,
                judge_provider=judge_provider,
                judge_quality_warning=quality_warning,
            )

        except asyncio.TimeoutError:
            log.warning("Judge call timeout after %ds", self.timeout_seconds)
            return None
        except Exception as exc:
            log.warning("Judge call exception: %s", exc)
            return None

    async def debate(
        self,
        question: str,
        user_id: int,
        chat_id: int,
    ) -> DebateResult:
        """Run a multi-AI debate.

        1. Identify available providers (with deduplication)
        2. Query all in parallel (asyncio.gather)
        3. Collect responses + errors
        4. Create consensus analysis

        Args:
            question: The user question.
            user_id: Telegram user ID.
            chat_id: Telegram chat ID.

        Returns:
            DebateResult with all responses, errors, and analysis.
        """
        t_start = time.monotonic()

        providers = self._select_providers()
        if not providers:
            return DebateResult(
                question=question,
                responses={},
                errors={"system": "No providers available"},
                consensus_analysis=None,
                duration_seconds=time.monotonic() - t_start,
                providers_queried=[],
            )

        log.info(
            "Debate started: %d providers (%s), question: %s",
            len(providers),
            providers,
            question[:80],
        )

        # Phase 1: query all providers in parallel
        t_providers_start = time.monotonic()
        tasks = [
            self._query_provider(name, question, user_id, chat_id) for name in providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        t_providers_elapsed = time.monotonic() - t_providers_start

        responses: dict[str, str] = {}
        errors: dict[str, str] = {}

        for result in results:
            if isinstance(result, Exception):
                errors["unknown"] = str(result)
                continue
            provider_name, text, error = result
            if text is not None:
                responses[provider_name] = text
            elif error is not None:
                errors[provider_name] = error

        log.info(
            "Debate phase 1 (provider calls): %.1fs, %d OK, %d errors",
            t_providers_elapsed,
            len(responses),
            len(errors),
        )

        # Consensus analysis
        consensus: str | None = None
        if responses:
            consensus = self._analyze_consensus(responses)

        # Phase 2: final review (LLM-as-Judge)
        verdict: FinalVerdict | None = None
        t_judge_elapsed: float = 0.0
        if len(responses) >= 2:
            t_judge_start = time.monotonic()
            verdict = await self.final_review(
                question=question,
                responses=responses,
                user_id=user_id,
                chat_id=chat_id,
            )
            t_judge_elapsed = time.monotonic() - t_judge_start
            log.info(
                "Debate phase 2 (judge call): %.1fs, verdict=%s",
                t_judge_elapsed,
                verdict.winner if verdict else "failed",
            )

        duration = time.monotonic() - t_start

        log.info(
            "Debate completed: %.1fs total (providers=%.1fs, judge=%.1fs), "
            "%d responses, %d errors",
            duration,
            t_providers_elapsed,
            t_judge_elapsed,
            len(responses),
            len(errors),
        )

        return DebateResult(
            question=question,
            responses=responses,
            errors=errors,
            consensus_analysis=consensus,
            final_verdict=verdict,
            duration_seconds=duration,
            providers_queried=providers,
        )
