from __future__ import annotations

import logging

from bmo_agent.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMNotConfiguredError(RuntimeError):
    pass


def get_anthropic_client(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.anthropic_configured:
        raise LLMNotConfiguredError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file (see .env.example) "
            "before running the agent -- see console.anthropic.com to create a key."
        )
    from anthropic import Anthropic

    return Anthropic(api_key=settings.anthropic_api_key)


def complete(
    system: str,
    user: str,
    settings: Settings | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    # Note: the installed anthropic SDK (2.3.0, Claude 5 family) no longer exposes
    # `temperature` as a top-level sampling control on messages.create(); classification
    # /extraction stability here instead comes from strict-JSON prompting plus
    # deterministic Python-side arithmetic (see agent/calculations.py), not from
    # temperature=0.
    settings = settings or get_settings()
    client = get_anthropic_client(settings)
    resp = client.messages.create(
        model=model or settings.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def complete_structured(
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict,
    settings: Settings | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> dict:
    """Force a structured JSON response via Anthropic tool-use instead of asking
    the model to write JSON in prose and regex/json.loads-parsing it. Categorically
    more reliable -- the SDK validates the shape before the response ever reaches
    us, instead of us hoping the model closes its brackets correctly. Replaces the
    parse_json_response() + "failed to parse JSON, defaulting to..." fallback
    pattern seen in classify/verify/extract_figures during benchmark testing.
    """
    settings = settings or get_settings()
    client = get_anthropic_client(settings)
    # Note: this model generation rejects tool_choice={"type": "tool", ...} ("not
    # supported for this model") -- forcing a specific tool call server-side isn't
    # available, so we use "auto" (reliable in practice when it's the only tool
    # offered) plus an explicit instruction, rather than silently hoping.
    # Extended thinking is on by default for this model generation and can consume
    # the entire max_tokens budget before the model ever emits the tool call --
    # confirmed during testing: `verify` and `extract_figures` silently got a
    # thinking-only response with no tool_use block at small max_tokens values.
    # thinking={"type": "disabled"} isn't accepted on this model ("the model does
    # not think before responding" -- per the API's own error message); the
    # equivalent for a tool-use flow is "between_tools", which skips the long
    # pre-answer reasoning pass these short structured calls don't need.
    system_with_instruction = f"{system}\n\nAlways respond by calling the `{tool_name}` tool."
    resp = client.messages.create(
        model=model or settings.anthropic_model,
        max_tokens=max_tokens,
        system=system_with_instruction,
        messages=[{"role": "user", "content": user}],
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }
        ],
        tool_choice={"type": "auto"},
        thinking={"type": "between_tools"},
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise ValueError(f"Model did not return a {tool_name!r} tool call; response: {resp.content!r}")
