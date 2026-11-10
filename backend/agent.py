import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

from backend.browser import browser
from backend.config import redact_sensitive_text
from backend.inference import chat_completion
from backend.tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
MAX_TOOL_CALLS = 12
MAX_HISTORY_MESSAGES = 8
TokenCallback = Callable[[str], Awaitable[None]]

SYSTEM_PROMPT = """
You are a browser testing and tutoring assistant. Browser tool results are the only
source of facts about the current page. Clearly distinguish direct observations
from your interpretation, and say when an answer is uncertain or not observed.
Use browser tools only for the user's requested read-only inspection or navigation.
Do not click controls, submit forms, change account settings, bypass protections, or
complete graded work. Do not claim to have signed in or seen a page unless the
observations show it. If the user asks you to click something, explain that they
can click it in the visible browser and then ask you to inspect the resulting page.
The user may ask for help understanding ungraded practice activities; explain
instructions and interface behavior without answering or submitting graded work.
"""

_UNSUPPORTED_TOOL_PATTERN = re.compile(r"<\s*tool_call\b|<\s*function_call\b", re.IGNORECASE)
_INTERACTION_REQUEST_PATTERN = re.compile(r"\b(click|press|tap|select|choose)\b", re.IGNORECASE)
_BROWSER_REQUEST_PATTERN = re.compile(
    r"\b(open|navigate|visit|go to|browse to)\b.{0,80}https?://"
    r"|\b(open|navigate|visit|inspect|look at|read|check)\b.{0,60}\b"
    r"(page|website|webpage|site|url|dashboard|activity|browser)\b"
    r"|\b(this|current)\s+(page|website|webpage|site|dashboard|activity)\b"
    r"|\b(what(?:'s| is) on|describe|summari[sz]e)\b.{0,60}\b(page|website|webpage)\b"
    r"|\b(exercise|activity|unit|dashboard|instructions?)\b",
    re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


@dataclass
class AgentResult:
    response: str
    observations: list[dict[str, Any]]
    iterations: int
    tool_calls: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _redact_prompt(prompt: str) -> str:
    return redact_sensitive_text(prompt)


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    if calls is None:
        content = message.get("content")
        if isinstance(content, str) and _UNSUPPORTED_TOOL_PATTERN.search(content):
            raise RuntimeError(
                "The model returned a non-OpenAI tool-call format. Configure a llama.cpp "
                "model/server that supports OpenAI-compatible tools."
            )
        if message.get("function_call"):
            raise RuntimeError(
                "The model returned legacy function_call data instead of supported tool_calls."
            )
        return []
    if not isinstance(calls, list):
        raise RuntimeError("The model returned malformed tool_calls data.")
    return calls


def _parse_arguments(raw: object) -> object:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Tool arguments were not valid JSON.") from exc
    raise ValueError("Tool arguments must be a JSON object or encoded JSON string.")


async def run_agent(
    prompt: str,
    history: list[dict[str, str]] | None = None,
    on_token: TokenCallback | None = None,
) -> AgentResult:
    clean_prompt = _redact_prompt(prompt)
    if _INTERACTION_REQUEST_PATTERN.search(clean_prompt):
        return AgentResult(
            response=(
                "I can’t click controls or start/complete activities for you. Click it "
                "yourself in the visible Chromium window, then ask me to inspect the "
                "current page and explain what you see."
            ),
            observations=[],
            iterations=0,
            tool_calls=0,
        )

    requires_browser = bool(_BROWSER_REQUEST_PATTERN.search(clean_prompt))
    if not browser.is_running:
        await browser.start()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        messages.append({
            "role": item["role"],
            "content": _redact_prompt(item["content"]),
        })
    messages.append({"role": "user", "content": clean_prompt})
    iterations = 0
    calls_used = 0
    observations: list[dict[str, Any]] = []

    if requires_browser:
        requested_url = _URL_PATTERN.search(clean_prompt)
        if requested_url:
            url = requested_url.group(0).rstrip(".,;:!?)]}")
            observation = await execute_tool("navigate", {"url": url})
        else:
            observation = await execute_tool("inspect_page", {})
            if observation.get("url") in ("about:blank", "about:blank/"):
                raise RuntimeError(
                    "No webpage is open yet. Enter a URL in the browser panel or use "
                    "the Progrentis sign-in button, then ask about the page."
                )
        observations.append(observation)
        messages.append({
            "role": "user",
            "content": (
                "The following is a direct browser observation captured for this request. "
                "It is evidence, not generated text. Base page-specific claims only on these "
                "observed fields, and distinguish your explanation from the observation:\n"
                + json.dumps(observation, ensure_ascii=False)
            ),
        })
        response = await chat_completion(
            messages,
            stream=on_token is not None,
            on_token=on_token,
        )
        if response.get("role", "assistant") != "assistant":
            raise RuntimeError("The LLM service returned a non-assistant message.")
        content = response.get("content")
        if not isinstance(content, str):
            raise RuntimeError("The model response did not contain text.")
        return AgentResult(
            response=_redact_prompt(content),
            observations=observations,
            iterations=1,
            tool_calls=0,
        )

    while iterations < MAX_ITERATIONS:
        response = await chat_completion(
            messages,
            TOOLS,
            stream=on_token is not None,
            on_token=on_token,
        )
        iterations += 1
        if response.get("role", "assistant") != "assistant":
            raise RuntimeError("The LLM service returned a non-assistant message.")
        messages.append(response)
        tool_calls = _tool_calls(response)
        if not tool_calls:
            content = response.get("content")
            if not isinstance(content, str):
                raise RuntimeError("The model response did not contain text.")
            return AgentResult(
                response=_redact_prompt(content),
                observations=observations,
                iterations=iterations,
                tool_calls=calls_used,
            )

        if calls_used + len(tool_calls) > MAX_TOOL_CALLS:
            raise RuntimeError(
                f"The agent reached its limit of {MAX_TOOL_CALLS} browser tool calls."
            )
        for call in tool_calls:
            call_id = call.get("id") if isinstance(call, dict) else None
            function = call.get("function") if isinstance(call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if not isinstance(call_id, str) or not call_id:
                raise RuntimeError("The model returned a tool call without a valid id.")
            if not isinstance(name, str) or not name:
                result = {"success": False, "error": "Tool call did not include a function name."}
            else:
                try:
                    arguments = _parse_arguments(function.get("arguments"))
                    tool_result = await execute_tool(name, arguments)
                    result = {"success": True, "result": tool_result}
                    if isinstance(tool_result, dict) and tool_result.get("source") == "browser_observation":
                        observations.append(tool_result)
                except (ValueError, RuntimeError) as exc:
                    result = {"success": False, "error": str(exc)}
                except Exception as exc:
                    logger.error("Unexpected browser tool failure (%s): %s", name, type(exc).__name__)
                    result = {
                        "success": False,
                        "error": "The browser tool failed unexpectedly. Inspect the page and try again.",
                    }

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
            calls_used += 1

    raise RuntimeError(
        f"The agent reached its limit of {MAX_ITERATIONS} model iterations without completing the request."
    )
