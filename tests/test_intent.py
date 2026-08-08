"""Tests for the intent classifier (Day 7), fully offline via ScriptedLLMClient."""

from __future__ import annotations

from copilot.agent.intent import (
    INTENT_LABELS,
    UNKNOWN_INTENT,
    _parse_intent_content,
    classify_intent,
)
from copilot.llm.base import ChatMessage, LLMProviderError, LLMResponse
from copilot.llm.providers import ScriptedLLMClient


def test_classify_intent_happy_path():
    llm = ScriptedLLMClient([LLMResponse(content='{"intent": "place_order", "confidence": 0.87}')])
    result = classify_intent("I want to order 2 notebooks", llm=llm)
    assert result.label == "place_order"
    assert result.confidence == 0.87
    assert result.low_confidence is False

    # the call was a plain classification prompt, not tool-calling
    call = llm.calls[0]
    assert call["tools"] is None
    assert [m.role for m in call["messages"]] == ["system", "user"]


def test_classify_intent_below_threshold_is_low_confidence():
    llm = ScriptedLLMClient([LLMResponse(content='{"intent": "faq_policy", "confidence": 0.3}')])
    result = classify_intent("hmm not sure what I want", llm=llm, confidence_threshold=0.5)
    assert result.label == "faq_policy"
    assert result.low_confidence is True


def test_classify_intent_tolerates_prose_around_json():
    llm = ScriptedLLMClient(
        [
            LLMResponse(
                content='sure:\n{"intent": "shipping_inquiry", "confidence": 0.7}\nhope that helps!'
            )
        ]
    )
    result = classify_intent("how much to ship to Berlin?", llm=llm)
    assert result.label == "shipping_inquiry"
    assert result.confidence == 0.7


def test_classify_intent_falls_back_to_unknown_on_malformed_reply():
    llm = ScriptedLLMClient([LLMResponse(content="I'm not sure how to classify that.")])
    result = classify_intent("???", llm=llm)
    assert result.label == UNKNOWN_INTENT
    assert result.confidence == 0.0
    assert result.low_confidence is True  # 0.0 is always below any real threshold


def test_classify_intent_falls_back_on_empty_content():
    # ScriptedLLMClient's own default when its queue is exhausted: no crash.
    llm = ScriptedLLMClient([])
    result = classify_intent("anything", llm=llm)
    assert result.label == UNKNOWN_INTENT
    assert result.low_confidence is True


def test_classify_intent_rejects_label_outside_taxonomy():
    llm = ScriptedLLMClient([LLMResponse(content='{"intent": "make_a_wish", "confidence": 0.99}')])
    result = classify_intent("I wish for a pony", llm=llm)
    assert result.label == UNKNOWN_INTENT
    assert result.confidence == 0.0


def test_classify_intent_normalizes_label_casing_and_separators():
    llm = ScriptedLLMClient([LLMResponse(content='{"intent": "Human Request", "confidence": 0.8}')])
    result = classify_intent("get me a person", llm=llm)
    assert result.label == "human_request"


def test_classify_intent_clamps_out_of_range_confidence():
    llm = ScriptedLLMClient([LLMResponse(content='{"intent": "greeting", "confidence": 5}')])
    result = classify_intent("hi!", llm=llm)
    assert result.confidence == 1.0


def test_parse_intent_content_defaults_confidence_when_unparseable():
    # a valid label with a garbled confidence field keeps the label but treats
    # confidence as unknown (0.0) rather than discarding the whole prediction.
    label, confidence = _parse_intent_content(
        '{"intent": "product_inquiry", "confidence": "not a number"}'
    )
    assert label == "product_inquiry"
    assert confidence == 0.0


def test_all_intent_labels_are_snake_case_and_unique():
    assert len(INTENT_LABELS) == len(set(INTENT_LABELS))
    assert all(label == label.lower() and " " not in label for label in INTENT_LABELS)


def test_classify_intent_history_is_not_required():
    # sanity: classify_intent only needs the message + llm, not conversation history
    llm = ScriptedLLMClient([LLMResponse(content='{"intent": "greeting", "confidence": 0.9}')])
    result = classify_intent("hello", llm=llm)
    assert result.label == "greeting"
    assert isinstance(llm.calls[0]["messages"][1], ChatMessage)


class _UnreachableLLM:
    """A minimal LLMClient double that simulates the provider being down."""

    def chat(self, messages, tools=None, temperature=None):
        raise LLMProviderError("connection refused")


def test_classify_intent_degrades_gracefully_when_provider_is_unreachable():
    result = classify_intent("hello", llm=_UnreachableLLM())
    assert result.label == UNKNOWN_INTENT
    assert result.confidence == 0.0
    assert result.low_confidence is True
