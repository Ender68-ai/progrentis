import logging

from backend.config import settings


def configure_logging():
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOG_DIR / "agent.log"),
        ],
    )