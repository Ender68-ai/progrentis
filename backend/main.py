from contextlib import asynccontextmanager
from fastapi import FastAPI

from backend.config import settings
from backend.logging_config import configure_logging

from pydantic import BaseModel
from backend.inference import chat_completion
from backend.agent import run_agent

from backend.agent import run_agent

import logging

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.APP_NAME, settings.VERSION)

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.VERSION,
    }

class ModelRequest(BaseModel):
    prompt: str

class AgentRequest(BaseModel):
    prompt: str

@app.post("/agent/run")
async def agent_run(request: AgentRequest):
    try:
        result = await run_agent(request.prompt)
        return {"response": result}

    except Exception as exc:
        logger.exception("Agent execution failed")
        raise HTTPException(500, str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )