"""Provider-agnostic Claude client + a structured-review call with retries.

Auto-detects the provider from the environment, in priority order:
  1. Azure AI Foundry (Anthropic-compatible) — ``ANTHROPIC_BASE_URL`` + ``ANTHROPIC_API_KEY``.
  2. Foundry SDK helper — ``ANTHROPIC_FOUNDRY_API_KEY`` (+ ``ANTHROPIC_FOUNDRY_RESOURCE``).
  3. First-party Anthropic API — ``ANTHROPIC_API_KEY`` alone.
Secrets are read from the environment only (see code/.env / .env.example).
"""
from __future__ import annotations

import email.utils
import os
import random
import time

import anthropic

from . import config


def make_client():
    """Build an Anthropic client for whichever provider is configured.

    Our own backoff loop (``call_review``) owns retries, so the SDK's client-level
    retries are disabled (``max_retries=0``) to avoid compounding the two.
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if base_url and api_key:
        # Anthropic-compatible endpoint (e.g. Azure AI Foundry's /anthropic route).
        return anthropic.Anthropic(
            api_key=api_key, base_url=base_url,
            timeout=config.REQUEST_TIMEOUT, max_retries=0,
        )
    if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY"):
        return anthropic.AnthropicFoundry(
            resource=os.environ["ANTHROPIC_FOUNDRY_RESOURCE"],
            api_key=os.environ["ANTHROPIC_FOUNDRY_API_KEY"],
            timeout=config.REQUEST_TIMEOUT, max_retries=0,
        )
    if api_key:
        return anthropic.Anthropic(timeout=config.REQUEST_TIMEOUT, max_retries=0)
    raise RuntimeError(
        "No credentials found. Set ANTHROPIC_API_KEY (and ANTHROPIC_BASE_URL for an "
        "Azure AI Foundry / Anthropic-compatible endpoint), or ANTHROPIC_FOUNDRY_API_KEY "
        "(+ ANTHROPIC_FOUNDRY_RESOURCE)."
    )


def provider_name() -> str:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    if base_url:
        return "azure-foundry" if "azure" in base_url.lower() else "anthropic-compatible"
    if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY"):
        return "foundry"
    return "anthropic"


_RETRYABLE = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


def _usage_dict(usage) -> dict:
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }


def _request_kwargs(model, max_tokens, system_blocks, user_content, tool) -> dict:
    """Build messages.create kwargs, honoring config.THINKING_ENABLED.

    No ``temperature`` is ever sent (sampling params are rejected on Opus 4.8/4.7).
    Thinking OFF (default): force the tool (``tool_choice=tool``) for a guaranteed
    structured reply. Thinking ON: forcing a specific tool is disallowed with extended
    thinking, so use ``tool_choice=auto`` and extract the tool_use block from the reply.
    """
    kw = dict(
        model=model, max_tokens=max_tokens, system=system_blocks,
        messages=[{"role": "user", "content": user_content}], tools=[tool],
    )
    if config.THINKING_ENABLED:
        budget = max(1024, min(max_tokens - 512, max_tokens // 2))
        kw["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kw["tool_choice"] = {"type": "auto"}
    else:
        kw["tool_choice"] = {"type": "tool", "name": tool["name"]}
    return kw


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Respect Retry-After on 429; fall back to exponential backoff with jitter."""
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(exc, anthropic.RateLimitError):
        headers = getattr(resp, "headers", {}) or {}
        raw = headers.get("retry-after") or headers.get("retry-after-ms")
        if raw:
            try:
                # retry-after-ms takes priority
                if headers.get("retry-after-ms"):
                    after = float(raw) / 1000.0
                else:
                    after = float(raw)
            except ValueError:
                # HTTP-date format fallback
                try:
                    parsed = email.utils.parsedate_to_datetime(raw)
                    after = max(0.0, parsed.timestamp() - time.time())
                except Exception:
                    after = None
            if after is not None and 0 < after <= 60:
                return after
    return min(60.0, (2 ** attempt) + random.uniform(0, 1))


def call_review(client, model: str, system_blocks: list, user_content: list, tool: dict,
                max_tokens: int = config.MAX_TOKENS,
                max_retries: int = config.MAX_RETRIES) -> tuple[dict, dict]:
    """Make one ``submit_review`` call. Returns (perception, usage)."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                **_request_kwargs(model, max_tokens, system_blocks, user_content, tool)
            )
            for block in resp.content:
                if block.type == "tool_use" and block.name == tool["name"]:
                    return dict(block.input), _usage_dict(resp.usage)
            raise RuntimeError(f"No '{tool['name']}' tool_use block in response (stop={resp.stop_reason}).")
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = _retry_delay(exc, attempt)
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            # Retry only on server-side 5xx; surface 4xx (other than 429) immediately.
            if exc.status_code >= 500:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(_retry_delay(exc, attempt))
            else:
                raise
    raise RuntimeError(f"call_review exhausted retries: {last_exc!r}") from last_exc
