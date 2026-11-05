import json
import logging

from backend.inference import chat_completion
from backend.tools import TOOLS, execute_tool
from backend.browser import browser


logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = """
You are a browser testing and inspection assistant.

You can inspect webpages using the tools provided to you.

Rules:
- Only use the available tools.
- Never invent observations about a webpage.
- Base your answers on actual browser observations.
- Explain what you find clearly.
- Do not submit forms, make purchases, or change account settings.
- Ask for permission before performing actions that modify webpage state.
"""


async def run_agent(prompt: str):
    if not browser.page:
        await browser.start()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    tool_calls_used = 0

    while True:
        response = await chat_completion(messages, TOOLS)

        messages.append(response)

        tool_calls = response.get("tool_calls")

        if not tool_calls:
            return response.get("content", "")

        if tool_calls_used + len(tool_calls) > MAX_TOOL_CALLS:
            return "Maximum tool-call limit reached. Please narrow your request."

        for call in tool_calls:
            function = call["function"]
            name = function["name"]

            try:
                arguments = json.loads(function["arguments"])
                result = await execute_tool(name, arguments)

                tool_result = {
                    "success": True,
                    "result": result,
                }

            except Exception as exc:
                logger.exception("Tool execution failed: %s", name)

                tool_result = {
                    "success": False,
                    "error": str(exc),
                }

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(tool_result),
            })

            tool_calls_used += 1