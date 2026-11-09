import logging
from typing import Any

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)


def _completion_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


async def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    settings.require_llm()
    payload: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    if tools:
        payload.update(tools=tools, tool_choice="auto")

    try:
        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            response = await client.post(
                _completion_url(settings.LLM_BASE_URL),
                json=payload,
                headers={"Authorization": f"Bearer {settings.LLM_API_KEY}"},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"The LLM request timed out after {settings.LLM_TIMEOUT} seconds. "
            "Check the Kaggle runtime and Cloudflare tunnel."
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("LLM request returned HTTP %s", exc.response.status_code)
        raise RuntimeError(
            f"The LLM service returned HTTP {exc.response.status_code}. "
            "Check the Kaggle inference server and API configuration."
        ) from exc
    except httpx.RequestError as exc:
        logger.warning("LLM connection failed (%s)", type(exc).__name__)
        raise RuntimeError(
            "Could not connect to the LLM service. Check the Kaggle runtime, "
            "Cloudflare tunnel URL, and network connection."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("The LLM service returned invalid JSON.") from exc

    if not isinstance(data, dict):
        raise RuntimeError("The LLM service returned an unexpected completion response.")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError("The LLM service returned an unexpected completion response.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("The LLM service returned an invalid assistant message.")
    if tools and choice.get("finish_reason") == "tool_calls" and not message.get("tool_calls"):
        raise RuntimeError("The model indicated tool use but returned no supported tool calls.")
    return message
