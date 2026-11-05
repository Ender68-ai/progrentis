import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    APP_NAME = "Progrentis Agent"
    VERSION = "0.1.0"

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", "8000"))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    DATA_DIR = ROOT_DIR / "data"
    LOG_DIR = ROOT_DIR / "logs"
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "")
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))


settings = Settings()