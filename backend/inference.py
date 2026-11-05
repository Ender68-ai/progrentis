import httpx

from backend.config import settings


async def chat_completion(messages: list, tools: list | None = None):
    if not settings.LLM_BASE_URL:
        raise RuntimeError("LLM_BASE_URL is not configured")

    payload = {
        "model": settings.LLM_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
    }

    async with httpx.AsyncClient(
        timeout=settings.LLM_TIMEOUT
    ) as client:
        response = await client.post(
            f"{settings.LLM_BASE_URL.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )

        response.raise_for_status()
        data = response.json()

    return data["choices"][0]["message"]