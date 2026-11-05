from backend.browser import browser


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigate to a webpage and inspect it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The HTTP or HTTPS URL to visit."
                    }
                },
                "required": ["url"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_page",
            "description": "Inspect the currently open webpage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        }
    }
]


async def execute_tool(name: str, arguments: dict):
    if name == "navigate":
        url = arguments["url"]

        if not url.startswith(("http://", "https://")):
            raise ValueError("Only HTTP and HTTPS URLs are allowed.")

        return await browser.navigate(url)

    if name == "inspect_page":
        return await browser.inspect()

    raise ValueError(f"Unknown tool: {name}")