"""OpenRouter API client for escalation review.

OpenRouter exposes an OpenAI-compatible /v1/chat/completions endpoint that fronts
many models. We use it as the escalation reviewer for the research pipeline:
when the local Qwen research agent produces a brief, OpenRouter scores it
against a rubric before the brief is handed to the video-production pipeline.

Why escalation-only and not default:
- OpenRouter free tier has tight rate limits; using it for every stage burns
  budget fast.
- Local Ollama models handle JSON-shaped output fine for the high-volume steps.
- We only need a stronger model when scoring quality of a final artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests


# ---- Result types ---------------------------------------------------------


@dataclass(frozen=True)
class ChatUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ChatResponse:
    id: str
    model: str
    content: str
    finish_reason: str | None
    usage: ChatUsage


@dataclass(frozen=True)
class QualityScore:
    technical_accuracy: int
    depth: int
    uniqueness: int
    troubleshooting_value: int
    source_grounding: int
    ready_for_script: bool
    rationale: str

    @property
    def composite(self) -> float:
        return (
            self.technical_accuracy
            + self.depth
            + self.uniqueness
            + self.troubleshooting_value
            + self.source_grounding
        ) / 5.0


# ---- Errors ---------------------------------------------------------------


class OpenRouterError(RuntimeError):
    """Raised on any OpenRouter API failure (HTTP, parse, schema)."""


# ---- Client ---------------------------------------------------------------


# Default JSON schema for review_quality() — matches the doc's quality_score.json spec.
QUALITY_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "technical_accuracy": {"type": "integer", "minimum": 1, "maximum": 10},
        "depth": {"type": "integer", "minimum": 1, "maximum": 10},
        "uniqueness": {"type": "integer", "minimum": 1, "maximum": 10},
        "troubleshooting_value": {"type": "integer", "minimum": 1, "maximum": 10},
        "source_grounding": {"type": "integer", "minimum": 1, "maximum": 10},
        "ready_for_script": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": [
        "technical_accuracy",
        "depth",
        "uniqueness",
        "troubleshooting_value",
        "source_grounding",
        "ready_for_script",
        "rationale",
    ],
    "additionalProperties": False,
}


class OpenRouterClient:
    """Thin client for the OpenRouter /v1/chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 120.0,
        referer: str = "https://knowledge-pipeline.local",
        app_title: str = "knowledge-pipeline",
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter api_key must not be empty")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._referer = referer
        self._app_title = app_title

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._referer,
            "X-Title": self._app_title,
        }

    # ---- Transport --------------------------------------------------------

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = requests.post(
                url,
                json=body,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"HTTP error calling {url}: {exc}") from exc

        if response.status_code >= 400:
            # Try to surface OpenRouter's error message.
            try:
                err_body = response.json()
                if isinstance(err_body, dict) and "error" in err_body:
                    err = err_body["error"]
                    if isinstance(err, dict):
                        message = err.get("message", response.text)
                    else:
                        message = str(err)
                else:
                    message = response.text
            except ValueError:
                message = response.text
            raise OpenRouterError(
                f"OpenRouter API {response.status_code} on {url}: {message}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise OpenRouterError(f"Non-JSON response from {url}: {exc}") from exc

    # ---- Public API -------------------------------------------------------

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Single-turn chat completion.

        Args:
            model: OpenRouter model id, e.g. "anthropic/claude-sonnet-4-20250514".
            messages: list of {"role": "user"|"assistant"|"system", "content": "..."}.
            temperature: 0.0 deterministic, up to 2.0 for creative.
            max_tokens: optional cap on completion length.
            response_format: optional JSON schema spec for structured output.
                Format: {"type": "json_schema", "json_schema": {...}}.
                Pass {"type": "json_object"} to force JSON without a schema.

        Returns:
            ChatResponse with the model's content and token usage.

        Raises:
            ValueError: on invalid messages or model.
            OpenRouterError: on HTTP or parse failures.
        """
        if not model:
            raise ValueError("model must not be empty")
        if not messages:
            raise ValueError("messages must contain at least one message")
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("each message must have 'role' and 'content'")

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = response_format

        raw = self._post("/chat/completions", body)
        return _parse_chat_response(raw)

    def review_quality(
        self,
        *,
        brief_text: str,
        topic: str,
        model: str,
    ) -> QualityScore:
        """Score a research brief against the channel's quality rubric.

        Uses `json_object` response_format (free-tier models on OpenRouter
        reject `json_schema` as 400) and validates the shape locally via
        `_parse_quality_score`. Returns a QualityScore with per-criterion 1-10
        ratings and a composite property that averages them.

        Args:
            brief_text: the research_brief.md content to score.
            topic: the original research topic, for context.
            model: OpenRouter model id to use as the reviewer.

        Raises:
            OpenRouterError: on HTTP, parse, or schema-validation failures.
        """
        system_prompt = (
            "You are a strict technical reviewer for an internals-focused YouTube channel. "
            "Score the candidate research brief on the dimensions below. "
            "Be honest: a 9 or 10 means genuinely exceptional, not 'competent'. "
            "Use the full 1-10 range; do not cluster scores around 7-8."
        )
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Research brief to review:\n---\n{brief_text}\n---\n\n"
            "Score it on:\n"
            "- technical_accuracy: are the claims correct and verifiable?\n"
            "- depth: does it cover internals (packet flow, code paths, failure modes) rather than surface-level definitions?\n"
            "- uniqueness: does it cover material that typical YouTube videos on this topic miss?\n"
            "- troubleshooting_value: does it give the reader concrete debugging / verification steps?\n"
            "- source_grounding: are the claims tied to specific RFCs, docs, or source-code paths?\n\n"
            "ready_for_script=true only if all five dimensions are 7 or higher.\n"
            "rationale: one paragraph naming the brief's main strength and the single biggest improvement that would unlock a higher score."
        )
        raw = self.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return _parse_quality_score(raw.content)


# ---- Response parsing -----------------------------------------------------


def _parse_chat_response(raw: dict[str, Any]) -> ChatResponse:
    choices = raw.get("choices") or []
    if not choices:
        raise OpenRouterError(f"OpenRouter response had no choices: {raw}")
    first = choices[0]
    if not isinstance(first, dict):
        raise OpenRouterError(f"OpenRouter response had malformed choice: {first}")
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        raise OpenRouterError(f"OpenRouter response had no message.content: {first}")
    usage_raw = raw.get("usage") or {}
    usage = ChatUsage(
        prompt_tokens=_safe_int(usage_raw.get("prompt_tokens")),
        completion_tokens=_safe_int(usage_raw.get("completion_tokens")),
        total_tokens=_safe_int(usage_raw.get("total_tokens")),
    )
    return ChatResponse(
        id=str(raw.get("id", "")),
        model=str(raw.get("model", "")),
        content=str(content),
        finish_reason=first.get("finish_reason"),
        usage=usage,
    )


def _parse_quality_score(content: str) -> QualityScore:
    import json

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenRouterError(f"review_quality: model returned non-JSON content: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenRouterError(f"review_quality: expected object, got {type(data).__name__}")

    required = (
        "technical_accuracy",
        "depth",
        "uniqueness",
        "troubleshooting_value",
        "source_grounding",
        "ready_for_script",
        "rationale",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise OpenRouterError(f"review_quality: response missing fields: {missing}")

    for field_name in (
        "technical_accuracy",
        "depth",
        "uniqueness",
        "troubleshooting_value",
        "source_grounding",
    ):
        value = data[field_name]
        if not isinstance(value, int) or not 1 <= value <= 10:
            raise OpenRouterError(
                f"review_quality: {field_name} must be int in 1..10, got {value!r}"
            )

    if not isinstance(data["ready_for_script"], bool):
        raise OpenRouterError(
            f"review_quality: ready_for_script must be bool, got {data['ready_for_script']!r}"
        )

    return QualityScore(
        technical_accuracy=data["technical_accuracy"],
        depth=data["depth"],
        uniqueness=data["uniqueness"],
        troubleshooting_value=data["troubleshooting_value"],
        source_grounding=data["source_grounding"],
        ready_for_script=data["ready_for_script"],
        rationale=str(data["rationale"]).strip(),
    )


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
