from urllib.parse import urlsplit

from backend.browser import browser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Open a user-requested HTTP or HTTPS webpage and report observed page content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "An absolute HTTP or HTTPS URL explicitly requested by the user.",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_page",
            "description": "Read visible content and structure from the currently open webpage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def validate_tool_arguments(name: str, arguments: object) -> dict:
    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {name} must be a JSON object.")

    if name == "navigate":
        if set(arguments) != {"url"}:
            raise ValueError("navigate requires exactly one argument: url.")
        url = arguments["url"]
        if not isinstance(url, str) or not url.strip():
            raise ValueError("navigate.url must be a non-empty string.")
        try:
            parts = urlsplit(url.strip())
            host = parts.hostname
            parts.port
        except ValueError as exc:
            raise ValueError("navigate.url must be a valid absolute HTTP or HTTPS URL.") from exc
        if (
            parts.scheme not in ("http", "https")
            or not host
            or any(character.isspace() for character in parts.netloc)
        ):
            raise ValueError("navigate.url must be an absolute HTTP or HTTPS URL.")
        if parts.username or parts.password:
            raise ValueError("navigate.url must not contain embedded credentials.")
        return {"url": url.strip()}

    if name == "inspect_page":
        if arguments:
            raise ValueError("inspect_page does not accept arguments.")
        return {}

    raise ValueError(f"Unknown tool: {name}")


async def execute_tool(name: str, arguments: object) -> dict:
    validated = validate_tool_arguments(name, arguments)
    if name == "navigate":
        return await browser.navigate(validated["url"])
    return await browser.inspect()
