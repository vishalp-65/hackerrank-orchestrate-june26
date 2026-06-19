"""Provider-agnostic Claude client + a forced-tool-use review call with retries.

Auto-detects the provider from the environment: Azure AI Foundry when
``ANTHROPIC_FOUNDRY_API_KEY`` is set, otherwise the first-party API
(``ANTHROPIC_API_KEY``). Secrets are read from the environment only.
"""
from __future__ import annotations

import os
import random
import time

import anthropic

from . import config


def make_client():
    """Build an Anthropic client for whichever provider is configured."""
    if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY"):
        return anthropic.AnthropicFoundry(
            resource=os.environ["ANTHROPIC_FOUNDRY_RESOURCE"],
            api_key=os.environ["ANTHROPIC_FOUNDRY_API_KEY"],
            timeout=config.REQUEST_TIMEOUT,
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic(timeout=config.REQUEST_TIMEOUT)
    raise RuntimeError(
        "No credentials found. Set ANTHROPIC_FOUNDRY_API_KEY (+ ANTHROPIC_FOUNDRY_RESOURCE) "
        "or ANTHROPIC_API_KEY in the environment."
    )


def provider_name() -> str:
    return "foundry" if os.environ.get("ANTHROPIC_FOUNDRY_API_KEY") else "anthropic"


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


def call_review(client, model: str, system_blocks: list, user_content: list, tool: dict,
                max_tokens: int = config.MAX_TOKENS,
                max_retries: int = config.MAX_RETRIES) -> tuple[dict, dict]:
    """Make one forced ``submit_review`` call. Returns (perception, usage).

    No ``temperature`` and no ``thinking`` are sent: sampling params are rejected on
    Opus 4.8/4.7, and forced tool use is incompatible with extended thinking.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_content}],
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )
            for block in resp.content:
                if block.type == "tool_use" and block.name == tool["name"]:
                    return dict(block.input), _usage_dict(resp.usage)
            raise RuntimeError(f"No '{tool['name']}' tool_use block in response (stop={resp.stop_reason}).")
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = min(60.0, (2 ** attempt) + random.uniform(0, 1))
            time.sleep(delay)
        except anthropic.APIStatusError as exc:
            # Retry only on server-side 5xx; surface 4xx (other than 429) immediately.
            if exc.status_code >= 500:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(min(60.0, (2 ** attempt) + random.uniform(0, 1)))
            else:
                raise
    raise RuntimeError(f"call_review exhausted retries: {last_exc!r}") from last_exc
