"""Smoke tests for the OpenRouter client (run directly; no pytest).

Usage:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_openrouter_client.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.research.openrouter_client import (  # noqa: E402
    ChatResponse,
    OpenRouterClient,
    OpenRouterError,
    QualityScore,
    QUALITY_SCORE_SCHEMA,
    _parse_chat_response,
    _parse_quality_score,
)


def test_client_construction() -> None:
    c = OpenRouterClient("sk-or-v1-test")
    assert c._api_key == "sk-or-v1-test"
    assert c._base_url == "https://openrouter.ai/api/v1"

    try:
        OpenRouterClient("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty api_key")

    c2 = OpenRouterClient("tok", base_url="https://openrouter.ai/api/v1/")
    assert c2._base_url == "https://openrouter.ai/api/v1"


def test_parse_chat_response_full() -> None:
    raw = {
        "id": "gen-abc123",
        "model": "anthropic/claude-sonnet-4-20250514",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "## Research brief\n..."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
    }
    parsed = _parse_chat_response(raw)
    assert isinstance(parsed, ChatResponse)
    assert parsed.id == "gen-abc123"
    assert parsed.model == "anthropic/claude-sonnet-4-20250514"
    assert parsed.content.startswith("## Research brief")
    assert parsed.finish_reason == "stop"
    assert parsed.usage.prompt_tokens == 1000
    assert parsed.usage.completion_tokens == 500
    assert parsed.usage.total_tokens == 1500


def test_parse_chat_response_no_usage() -> None:
    raw = {
        "id": "gen-xyz",
        "model": "anthropic/claude-sonnet-4-20250514",
        "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    parsed = _parse_chat_response(raw)
    assert parsed.usage.prompt_tokens is None
    assert parsed.usage.completion_tokens is None
    assert parsed.usage.total_tokens is None


def test_parse_chat_response_errors() -> None:
    # No choices.
    try:
        _parse_chat_response({"id": "x", "model": "m", "choices": []})
    except OpenRouterError as e:
        assert "choices" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on no choices")

    # Malformed choice.
    try:
        _parse_chat_response({"id": "x", "model": "m", "choices": ["not-a-dict"]})
    except OpenRouterError as e:
        assert "malformed" in str(e) or "choice" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on malformed choice")

    # No message.content.
    try:
        _parse_chat_response({"id": "x", "model": "m", "choices": [{"message": {}}]})
    except OpenRouterError as e:
        assert "content" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on missing content")


def test_quality_score_schema_shape() -> None:
    # Schema must declare the right keys with the right types, otherwise the
    # remote model will produce output _parse_quality_score cannot ingest.
    props = QUALITY_SCORE_SCHEMA["properties"]
    for key in ("technical_accuracy", "depth", "uniqueness", "troubleshooting_value", "source_grounding"):
        assert props[key]["type"] == "integer"
        assert props[key]["minimum"] == 1
        assert props[key]["maximum"] == 10
    assert props["ready_for_script"]["type"] == "boolean"
    assert set(QUALITY_SCORE_SCHEMA["required"]) == {
        "technical_accuracy", "depth", "uniqueness", "troubleshooting_value",
        "source_grounding", "ready_for_script", "rationale",
    }


def test_parse_quality_score_valid() -> None:
    payload = (
        '{"technical_accuracy": 8, "depth": 9, "uniqueness": 7,'
        ' "troubleshooting_value": 6, "source_grounding": 9,'
        ' "ready_for_script": false,'
        ' "rationale": "Strong on internals but lacks concrete verification commands."}'
    )
    score = _parse_quality_score(payload)
    assert isinstance(score, QualityScore)
    assert score.technical_accuracy == 8
    assert score.depth == 9
    assert score.uniqueness == 7
    assert score.troubleshooting_value == 6
    assert score.source_grounding == 9
    assert score.ready_for_script is False
    assert "Strong on internals" in score.rationale
    # Composite is the simple mean of the five integer dimensions.
    assert abs(score.composite - 7.8) < 1e-9


def test_parse_quality_score_ready_true() -> None:
    payload = (
        '{"technical_accuracy": 9, "depth": 9, "uniqueness": 8,'
        ' "troubleshooting_value": 8, "source_grounding": 9,'
        ' "ready_for_script": true, "rationale": "Ready."}'
    )
    score = _parse_quality_score(payload)
    assert score.ready_for_script is True
    assert score.composite == 8.6


def test_parse_quality_score_errors() -> None:
    # Non-JSON content.
    try:
        _parse_quality_score("not json at all")
    except OpenRouterError as e:
        assert "non-JSON" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on non-JSON content")

    # Missing fields.
    try:
        _parse_quality_score('{"technical_accuracy": 5}')
    except OpenRouterError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on missing fields")

    # Out-of-range score.
    try:
        _parse_quality_score(
            '{"technical_accuracy": 11, "depth": 5, "uniqueness": 5,'
            ' "troubleshooting_value": 5, "source_grounding": 5,'
            ' "ready_for_script": false, "rationale": "x"}'
        )
    except OpenRouterError as e:
        assert "1..10" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on out-of-range score")

    # Non-bool ready_for_script.
    try:
        _parse_quality_score(
            '{"technical_accuracy": 5, "depth": 5, "uniqueness": 5,'
            ' "troubleshooting_value": 5, "source_grounding": 5,'
            ' "ready_for_script": "yes", "rationale": "x"}'
        )
    except OpenRouterError as e:
        assert "ready_for_script" in str(e)
    else:
        raise AssertionError("expected OpenRouterError on non-bool ready_for_script")


def test_chat_input_validation() -> None:
    c = OpenRouterClient("tok")

    # Empty model.
    try:
        c.chat(model="", messages=[{"role": "user", "content": "x"}])
    except ValueError as e:
        assert "model" in str(e)
    else:
        raise AssertionError("expected ValueError on empty model")

    # Empty messages.
    try:
        c.chat(model="m", messages=[])
    except ValueError as e:
        assert "messages" in str(e)
    else:
        raise AssertionError("expected ValueError on empty messages")

    # Malformed message.
    try:
        c.chat(model="m", messages=[{"role": "user"}])
    except ValueError as e:
        assert "content" in str(e)
    else:
        raise AssertionError("expected ValueError on message missing content")


TESTS = [
    test_client_construction,
    test_parse_chat_response_full,
    test_parse_chat_response_no_usage,
    test_parse_chat_response_errors,
    test_quality_score_schema_shape,
    test_parse_quality_score_valid,
    test_parse_quality_score_ready_true,
    test_parse_quality_score_errors,
    test_chat_input_validation,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            print(f"FAIL  {test.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {test.__name__}: {exc!r}")
            failed += 1
        else:
            print(f"OK    {test.__name__}")
    if failed:
        print(f"\n{failed} test(s) failed")
        return 1
    print("\nall smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
