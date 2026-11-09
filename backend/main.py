import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.agent import run_agent
from backend.browser import BrowserError, browser
from backend.config import (
    PROGRENTIS_LOGIN_URL,
    ConfigurationError,
    redact_sensitive_text,
    settings,
)
from backend.logging_config import configure_logging
from backend.tools import validate_tool_arguments

configure_logging()
logger = logging.getLogger(__name__)
WEB_PAGE = Path(__file__).parent / "web" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.VERSION)
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    app.state.agent_task = None
    app.state.last_error = None
    try:
        yield
    finally:
        task = app.state.agent_task
        if task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await browser.stop()
        logger.info("Shut down %s", settings.APP_NAME)


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)


class AgentRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000)


class NavigateRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class LoginRequest(BaseModel):
    authorized_origin: str = Field(min_length=1, max_length=512)


def _ensure_agent_idle(request: Request) -> None:
    task = getattr(request.app.state, "agent_task", None)
    if task and not task.done():
        raise HTTPException(409, "Stop the active agent run before controlling the browser.")


@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": settings.VERSION, "status": "running"}


@app.get("/app", include_in_schema=False)
async def app_page():
    return FileResponse(WEB_PAGE)


@app.get("/health")
async def health():
    return {"status": "healthy", "version": settings.VERSION}


@app.get("/status")
async def status(request: Request):
    observation = browser.last_inspection
    task = getattr(request.app.state, "agent_task", None)
    return {
        "browser_running": browser.is_running,
        "current_url": observation.get("url") if observation else None,
        "page_title": observation.get("title") if observation else None,
        "inspection": observation,
        "agent_running": bool(task and not task.done()),
        "last_error": getattr(request.app.state, "last_error", None),
        "llm_configured": bool(
            settings.LLM_BASE_URL and settings.LLM_API_KEY and settings.LLM_MODEL
        ),
        "progrentis_configured": bool(
            settings.PROGRENTIS_USERNAME and settings.PROGRENTIS_PASSWORD
        ),
    }


@app.post("/browser/start")
async def start_browser():
    try:
        await browser.start()
    except BrowserError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"browser_running": True}


@app.post("/browser/stop")
async def stop_browser(request: Request):
    task = getattr(request.app.state, "agent_task", None)
    if task and not task.done():
        raise HTTPException(409, "Stop the active agent run before closing the browser.")
    try:
        await browser.stop()
    except BrowserError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"browser_running": False}


@app.post("/browser/navigate")
async def navigate(payload: NavigateRequest, request: Request):
    _ensure_agent_idle(request)
    try:
        validated = validate_tool_arguments("navigate", {"url": payload.url})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        if not browser.is_running:
            await browser.start()
        observation = await browser.navigate(validated["url"])
    except (BrowserError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"observation": observation}


@app.post("/browser/inspect")
async def inspect_browser(request: Request):
    _ensure_agent_idle(request)
    try:
        return {"observation": await browser.inspect()}
    except BrowserError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/progrentis/login")
async def progrentis_login(payload: LoginRequest, request: Request):
    _ensure_agent_idle(request)
    try:
        result = await browser.authenticate_progrentis(payload.authorized_origin)
    except ConfigurationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except BrowserError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return result


@app.post("/progrentis/open-login")
async def open_progrentis_login(request: Request):
    _ensure_agent_idle(request)
    try:
        if not browser.is_running:
            await browser.start()
        observation = await browser.navigate(PROGRENTIS_LOGIN_URL)
    except (BrowserError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"observation": observation}


@app.post("/agent/run")
async def agent_run(payload: AgentRequest, request: Request):
    if not payload.prompt.strip():
        raise HTTPException(422, "Prompt cannot be blank.")
    if getattr(request.app.state, "agent_task", None) and not request.app.state.agent_task.done():
        raise HTTPException(409, "An agent run is already active.")

    request.app.state.last_error = None
    task = asyncio.create_task(run_agent(payload.prompt))
    request.app.state.agent_task = task
    try:
        result = await task
        return result.as_dict()
    except asyncio.CancelledError:
        return {"status": "stopped", "response": "The active run was stopped."}
    except Exception as exc:
        safe_error = redact_sensitive_text(str(exc))
        request.app.state.last_error = safe_error
        logger.error("Agent run failed (%s)", type(exc).__name__)
        raise HTTPException(502, safe_error) from exc
    finally:
        if request.app.state.agent_task is task:
            request.app.state.agent_task = None


@app.post("/agent/stop")
async def stop_agent(request: Request):
    task = getattr(request.app.state, "agent_task", None)
    if not task or task.done():
        return {"status": "idle"}
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return {"status": "stopped"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=False)
