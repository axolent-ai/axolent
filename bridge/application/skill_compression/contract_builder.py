"""Contract Builder: builds SkillContract drafts from /learn free-text.

Responsibilities:
  - Extract trigger phrase and instruction from natural-language /learn text
  - Stopword rejection (common words that must not become triggers)
  - needs_input state when extraction is uncertain
  - Origin = local_learn, permissions = deny-by-default

Anti-Regex-Schuld (Codex R5): When trigger and instruction cannot be
safely extracted, the builder returns needs_input status and does NOT
guess. The user is asked explicitly.

Dependencies: Python stdlib only (re).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from application.skill_compression.skill_contract import (
    ActivationConfig,
    ExecutionConfig,
    LifecycleConfig,
    SkillContract,
    new_skill_id,
    now_iso,
)


# ---------------------------------------------------------------
# Trigger stopwords (must NEVER become a skill trigger)
# ---------------------------------------------------------------

TRIGGER_STOPWORDS: frozenset[str] = frozenset(
    {
        # Conversational
        "ja",
        "nein",
        "ok",
        "okay",
        "gut",
        "danke",
        "bitte",
        "yes",
        "no",
        "thanks",
        "please",
        # Greetings
        "hi",
        "hallo",
        "hey",
        "bye",
        "ciao",
        # Commands (must never shadow bot commands)
        "/learn",
        "/forget",
        "/skills",
        "/installskill",
        "/help",
        "/start",
        "/settings",
        "/memory",
        "/remember",
        "/new",
        "/reset",
        "/stop",
        "/save",
        "/bookmarks",
        "/debate",
        "/setmodel",
        "/resetmodel",
        "/models",
        "/setlimit",
        "/usage",
        "/onboarding",
        "/lang",
        "/explain",
        "/import",
        "/skill",
    }
)

# Minimum / maximum trigger length
MIN_TRIGGER_LEN = 2
MAX_TRIGGER_LEN = 100


# ---------------------------------------------------------------
# Result types
# ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriggerValidationResult:
    """Outcome of trigger phrase validation."""

    valid: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Outcome of ContractBuilder.build().

    Attributes:
        contract: The built SkillContract (or minimal stub for needs_input).
        status: 'pending' if ready for preview, 'needs_input' if user must
                provide missing information.
        needs_input_reason: Human-readable prompt for what is missing.
    """

    contract: SkillContract
    status: str  # "pending" | "needs_input"
    needs_input_reason: str = ""


# ---------------------------------------------------------------
# Trigger validation
# ---------------------------------------------------------------


def validate_trigger(phrase: str) -> TriggerValidationResult:
    """Check if a trigger phrase is safe to use.

    Rejects:
      - Stopwords (ja, nein, ok, commands, etc.)
      - Too short (< 2 characters)
      - Too long (> 100 characters)
      - Starts with / (command-like)

    Args:
        phrase: The trigger phrase to validate.

    Returns:
        TriggerValidationResult with valid flag and reason.
    """
    normalized = phrase.strip().lower()

    if not normalized:
        return TriggerValidationResult(valid=False, reason="Trigger is empty.")

    if len(normalized) < MIN_TRIGGER_LEN:
        return TriggerValidationResult(
            valid=False,
            reason=f"Trigger too short (min {MIN_TRIGGER_LEN} characters).",
        )

    if len(normalized) > MAX_TRIGGER_LEN:
        return TriggerValidationResult(
            valid=False,
            reason=f"Trigger too long (max {MAX_TRIGGER_LEN} characters).",
        )

    if normalized in TRIGGER_STOPWORDS:
        return TriggerValidationResult(
            valid=False,
            reason=f"'{phrase}' is a reserved word and cannot be used as a trigger.",
        )

    if normalized.startswith("/"):
        return TriggerValidationResult(
            valid=False,
            reason="Trigger must not start with '/' (command conflict).",
        )

    return TriggerValidationResult(valid=True)


# ---------------------------------------------------------------
# Extraction patterns (DE + EN)
# ---------------------------------------------------------------

# DE patterns: "wenn ich X sage/schreibe" and reversed
_EXTRACT_PATTERNS_DE: list[re.Pattern[str]] = [
    # "wenn ich <TRIGGER> sage/schreibe/..., <INSTRUCTION>"
    re.compile(
        r"wenn\s+ich\s+(.+?)\s+(?:sage|schreibe|tippe|eingebe|sende)\s*[,.]?\s*(.+)",
        re.IGNORECASE,
    ),
    # "wenn ich sage/schreibe/... <TRIGGER>, <INSTRUCTION>"
    re.compile(
        r"wenn\s+ich\s+(?:sage|schreibe|tippe|eingebe|sende)\s+"
        r"([^,.\n]+?)\s*[,.]\s*(.+)",
        re.IGNORECASE,
    ),
    # "sobald ich <TRIGGER> sage/schreibe, <INSTRUCTION>"
    re.compile(
        r"sobald\s+ich\s+(.+?)\s+(?:sage|schreibe|tippe|eingebe|sende)\s*[,.]?\s*(.+)",
        re.IGNORECASE,
    ),
]

# EN patterns: "when I say/type/write X, <INSTRUCTION>"
_EXTRACT_PATTERNS_EN: list[re.Pattern[str]] = [
    # "when I say <TRIGGER>, <INSTRUCTION>"
    re.compile(
        r"when\s+I\s+(?:say|type|write|send|enter)\s+"
        r"([^,.\n]+?)\s*[,.]\s*(.+)",
        re.IGNORECASE,
    ),
]


def _extract_trigger_and_instruction(
    text: str,
) -> tuple[Optional[str], Optional[str]]:
    """Extract trigger phrase and instruction from /learn text.

    Tries DE patterns first, then EN. Returns (trigger, instruction)
    or (None, None) if extraction fails.

    Args:
        text: Raw /learn text (without the /learn prefix).

    Returns:
        Tuple of (trigger, instruction) or (None, None).
    """
    all_patterns = _EXTRACT_PATTERNS_DE + _EXTRACT_PATTERNS_EN

    for pattern in all_patterns:
        match = pattern.search(text)
        if match:
            trigger = match.group(1).strip().strip("\"'").strip()
            instruction = match.group(2).strip()
            if trigger and instruction:
                return trigger, instruction

    return None, None


def _derive_name(trigger: str, instruction: str) -> str:
    """Derive a human-readable skill name from trigger + instruction.

    Format: "<Trigger>: <first 40 chars of instruction>"
    Capitalized trigger.
    """
    trigger_part = trigger.capitalize()
    instr_part = instruction[:40].rstrip()
    if len(instruction) > 40:
        instr_part += "..."
    return f"{trigger_part}: {instr_part}"


# ---------------------------------------------------------------
# ContractBuilder
# ---------------------------------------------------------------


class ContractBuilder:
    """Build SkillContract drafts from /learn free-text.

    Design principles:
      - When trigger AND instruction are safely extractable:
        build a draft with status='pending' (ready for preview)
      - When NOT safely extractable: return needs_input, ask the user
      - Stopword triggers are rejected (needs_input with reason)
      - origin=local_learn always
      - Permissions: deny-by-default (empty = sandbox)
    """

    @staticmethod
    def build(text: str) -> BuildResult:
        """Build a SkillContract draft from /learn free-text.

        Args:
            text: The raw text after /learn (e.g.,
                  "wenn ich weiss schreibe, antworte mit 3 Farben")

        Returns:
            BuildResult with contract and status.
        """
        trigger, instruction = _extract_trigger_and_instruction(text)

        # Case 1: Cannot extract trigger/instruction
        if trigger is None or instruction is None:
            # Could be a preference-style skill (e.g., "bitte sei immer nett")
            # => needs_input, ask user for trigger
            ts = now_iso()
            stub_contract = SkillContract(
                id=new_skill_id(),
                name="",
                created_at=ts,
                updated_at=ts,
                execution=ExecutionConfig(instruction=text),
                lifecycle=LifecycleConfig(status="needs_input"),
                origin="local_learn",
            )
            return BuildResult(
                contract=stub_contract,
                status="needs_input",
                needs_input_reason="trigger_missing",
            )

        # Case 2: Trigger extracted but is a stopword
        trigger_check = validate_trigger(trigger)
        if not trigger_check.valid:
            ts = now_iso()
            stub_contract = SkillContract(
                id=new_skill_id(),
                name="",
                created_at=ts,
                updated_at=ts,
                execution=ExecutionConfig(instruction=instruction),
                lifecycle=LifecycleConfig(status="needs_input"),
                origin="local_learn",
            )
            return BuildResult(
                contract=stub_contract,
                status="needs_input",
                needs_input_reason="trigger_rejected",
            )

        # Case 3: Trigger and instruction safely extracted
        name = _derive_name(trigger, instruction)
        ts = now_iso()

        contract = SkillContract(
            id=new_skill_id(),
            name=name,
            created_at=ts,
            updated_at=ts,
            activation=ActivationConfig(
                kind="shortcut",
                mode="exact_phrase",
                phrases=(trigger.lower(),),
                match_scope="whole_message",
            ),
            execution=ExecutionConfig(
                type="llm_instruction",
                instruction=instruction,
            ),
            lifecycle=LifecycleConfig(status="draft"),
            origin="local_learn",
        )

        return BuildResult(
            contract=contract,
            status="pending",
        )

    @staticmethod
    def apply_trigger_edit(
        contract: SkillContract, new_trigger: str
    ) -> tuple[SkillContract, Optional[str]]:
        """Apply a trigger edit to an existing contract draft.

        Validates the new trigger. Returns (updated_contract, error_reason).
        error_reason is None on success, or a string describing the problem.

        Args:
            contract: The current draft contract.
            new_trigger: The new trigger phrase.

        Returns:
            Tuple of (updated contract, error reason or None).
        """
        check = validate_trigger(new_trigger)
        if not check.valid:
            return contract, check.reason

        from dataclasses import replace

        updated = replace(
            contract,
            activation=ActivationConfig(
                kind="shortcut",
                mode="exact_phrase",
                phrases=(new_trigger.lower(),),
                match_scope=contract.activation.match_scope,
                normalization=contract.activation.normalization,
                conditions=contract.activation.conditions,
                cooldown_seconds=contract.activation.cooldown_seconds,
            ),
            name=_derive_name(new_trigger, contract.execution.instruction),
            updated_at=now_iso(),
        )
        return updated, None

    @staticmethod
    def apply_instruction_edit(
        contract: SkillContract, new_instruction: str
    ) -> tuple[SkillContract, Optional[str]]:
        """Apply an instruction edit to an existing contract draft.

        Returns (updated_contract, error_reason).
        error_reason is None on success.

        Args:
            contract: The current draft contract.
            new_instruction: The new instruction text.

        Returns:
            Tuple of (updated contract, error reason or None).
        """
        if not new_instruction or not new_instruction.strip():
            return contract, "Instruction must not be empty."

        from dataclasses import replace

        trigger = ""
        if contract.activation.phrases:
            trigger = contract.activation.phrases[0]

        updated = replace(
            contract,
            execution=ExecutionConfig(
                type="llm_instruction",
                instruction=new_instruction.strip(),
                timeout_seconds=contract.execution.timeout_seconds,
                max_tool_calls=contract.execution.max_tool_calls,
            ),
            name=_derive_name(trigger, new_instruction.strip())
            if trigger
            else contract.name,
            updated_at=now_iso(),
        )
        return updated, None
