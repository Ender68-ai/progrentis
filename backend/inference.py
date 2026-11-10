import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)
TokenCallback = Callable[[str], Awaitable[None]]


def _completion_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


async def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    stream: bool = False,
    on_token: TokenCallback | None = None,
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
    if stream:
        payload["stream"] = True

    try:
        timeout = (
            httpx.Timeout(settings.LLM_TIMEOUT, read=None)
            if stream
            else settings.LLM_TIMEOUT
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            if stream:
                message: dict[str, Any] = {"role": "assistant", "content": ""}
                streamed_tool_calls: dict[int, dict[str, Any]] = {}
                async with client.stream(
                    "POST",
                    _completion_url(settings.LLM_BASE_URL),
                    json=payload,
                    headers={"Authorization": f"******"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk_data = line[5:].strip()
                        if chunk_data == "[DONE]":
                            break
                        if not chunk_data:
                            continue
                        chunk = httpx.Response(200, text=chunk_data).json()
                        choices = chunk.get("choices")
                        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                            continue
                        choice = choices[0]
                        delta = choice.get("delta")
                        if isinstance(delta, dict):
                            if delta.get("role"):
                                message["role"] = delta["role"]
                            content = delta.get("content")
                            if isinstance(content, str):
                                message["content"] += content
                                if on_token:
                                    await on_token(content)
                            tool_deltas = delta.get("tool_calls")
                            if isinstance(tool_deltas, list):
                                for tool_delta in tool_deltas:
                                    if not isinstance(tool_delta, dict):
                                        continue
                                    index = tool_delta.get("index")
                                    if not isinstance(index, int):
                                        continue
                                    tool_call = streamed_tool_calls.setdefault(
                                        index,
                                        {
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        },
                                    )
                                    if isinstance(tool_delta.get("id"), str):
                                        tool_call["id"] += tool_delta["id"]
                                    function_delta = tool_delta.get("function")
                                    if isinstance(function_delta, dict):
                                        for field in ("name", "arguments"):
                                            value = function_delta.get(field)
                                            if isinstance(value, str):
                                                tool_call["function"][field] += value
                        if choice.get("finish_reason") == "tool_calls" and not streamed_tool_calls:
                            raise RuntimeError(
                                "The model indicated tool use but returned no supported tool calls."
                            )
                if streamed_tool_calls:
                    message["tool_calls"] = [
                        streamed_tool_calls[index] for index in sorted(streamed_tool_calls)
                    ]
                if tools and not message.get("tool_calls") and not message.get("content"):
                    raise RuntimeError("The LLM service returned an empty completion.")
                return message

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
