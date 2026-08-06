"""Pre-agent screening: jailbreak / out-of-scope / PII / abuse. [Day 9]

Rule-based on purpose (PROJECT_PLAN.md §6: "start rule-based, add an LLM
classifier in V1") -- fast, deterministic, and its false-positive rate is
directly measurable in the Day 10-11 eval harness, unlike an LLM judge would
be. Rules are deliberately narrow: a guardrail that blocks eagerly on benign
stationery-shopping questions is worse than one that misses a rare edge case
(PROJECT_PLAN.md §9, metric 5 reports both block rate *and* false-positive
rate for exactly this reason).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SAFE_REFUSAL = (
    "I can't help with that, but I'm happy to help you find products, check an "
    "order, or answer a store policy question."
)

_PII_REFUSAL = (
    "Please don't share card or payment details in chat -- Paperbloom never asks "
    "for that here. I can still help with your order otherwise."
)

_ABUSE_ACK = "I'm going to connect you with a member of our team who can help with this."


@dataclass(frozen=True)
class GuardrailResult:
    """The outcome of running a message through the input rules."""

    action: str  # allow | block | escalate
    rule: str
    detail: str = ""
    safe_reply: str | None = None


# Checked in this order; the first match wins. Abuse is checked first because
# it should escalate even if the message also happens to look out-of-scope.
_ABUSE_PATTERNS = [
    re.compile(r"\b(kill|hurt|attack)\s+you\b", re.I),
    re.compile(r"\byou\s*('re|are)\s+(stupid|useless|worthless|garbage|trash|an idiot)\b", re.I),
    re.compile(r"\bf+u+c+k+\s+you\b", re.I),
]

_JAILBREAK_PATTERNS = [
    re.compile(r"\bignore\s+(your|all|any|previous|the)\s+(instructions|rules|prompt)\b", re.I),
    re.compile(r"\b(reveal|print|show)\s+(your\s+)?(system\s+prompt|instructions)\b", re.I),
    re.compile(r"\bact\s+as\s+(if\s+you|a).*\bwithout\s+(restrictions|rules|limits)\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
]

_PROHIBITED_ADVICE_PATTERNS = [
    re.compile(r"\b(cure|treat|diagnos\w*)\s+my\b", re.I),
    re.compile(r"\bmy\s+medical\s+condition\b", re.I),
    re.compile(r"\b(legal|financial|tax|investment)\s+advice\b", re.I),
    re.compile(r"\bshould\s+i\s+(invest|sue|divorce)\b", re.I),
]

# A run of 13-19 digits (optionally grouped) looks like a card number. Word
# boundaries keep this from matching short order/product ids like PB-NB-001.
_CARD_NUMBER_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")

_OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\bwrite\s+(my|me\s+a)\s+(essay|homework|r[ée]sum[ée]|cover letter)\b", re.I),
    re.compile(r"\bwrite\s+(a\s+)?(poem|song|story)\s+about\b", re.I),
    re.compile(r"\bsolve\s+this\s+math\b", re.I),
]


def evaluate_input(message: str) -> GuardrailResult:
    """Run all input rules in priority order; return the first non-allow match."""
    text = message or ""

    for pattern in _ABUSE_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                action="escalate",
                rule="abuse_or_threat",
                detail=pattern.pattern,
                safe_reply=_ABUSE_ACK,
            )

    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                action="block",
                rule="prompt_injection",
                detail=pattern.pattern,
                safe_reply=_SAFE_REFUSAL,
            )

    for pattern in _PROHIBITED_ADVICE_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                action="block",
                rule="prohibited_advice",
                detail=pattern.pattern,
                safe_reply=_SAFE_REFUSAL,
            )

    if _CARD_NUMBER_RE.search(text):
        return GuardrailResult(
            action="block",
            rule="pii_over_collection",
            detail="card-like number",
            safe_reply=_PII_REFUSAL,
        )

    for pattern in _OUT_OF_SCOPE_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(
                action="block",
                rule="out_of_scope",
                detail=pattern.pattern,
                safe_reply=_SAFE_REFUSAL,
            )

    return GuardrailResult(action="allow", rule="none")
