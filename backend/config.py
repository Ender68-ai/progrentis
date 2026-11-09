import os
from html import escape
from pathlib import Path
from urllib.parse import quote, quote_plus, urlsplit

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
PROGRENTIS_LOGIN_URL = "https://prod.progrentis.com/signin/tabid/1543/Default.aspx"
PROGRENTIS_HOST = "prod.progrentis.com"


class ConfigurationError(RuntimeError):
    """Raised when a required application setting is missing or invalid."""


class Settings:
    APP_NAME = "Progrentis Agent"
    VERSION = "0.2.0"

    def __init__(self) -> None:
        self.HOST = os.getenv("HOST", "127.0.0.1")
        self.PORT = self._integer("PORT", 8000, minimum=1, maximum=65535)
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

        self.DATA_DIR = ROOT_DIR / "data"
        self.LOG_DIR = ROOT_DIR / "logs"
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
        self.LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
        self.LLM_MODEL = os.getenv("LLM_MODEL", "nanbeige-4.3-33b").strip()
        self.LLM_TIMEOUT = self._integer("LLM_TIMEOUT", 300, minimum=1)
        self.PROGRENTIS_USERNAME = os.getenv("PROGRENTIS_USERNAME", "")
        self.PROGRENTIS_PASSWORD = os.getenv("PROGRENTIS_PASSWORD", "")

    @staticmethod
    def _integer(name: str, default: int, *, minimum: int, maximum: int | None = None) -> int:
        raw_value = os.getenv(name)
        try:
            value = default if raw_value is None else int(raw_value)
        except ValueError as exc:
            raise ConfigurationError(f"{name} must be an integer.") from exc
        if value < minimum or (maximum is not None and value > maximum):
            limit = f" between {minimum} and {maximum}" if maximum is not None else f" of at least {minimum}"
            raise ConfigurationError(f"{name} must be{limit}.")
        return value

    def require_llm(self) -> None:
        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", self.LLM_BASE_URL),
                ("LLM_API_KEY", self.LLM_API_KEY),
                ("LLM_MODEL", self.LLM_MODEL),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                f"Missing required LLM configuration: {', '.join(missing)}. "
                "Set these values in the project-root .env file."
            )
        try:
            base_url = urlsplit(self.LLM_BASE_URL)
            host = base_url.hostname
            base_url.port
        except ValueError as exc:
            raise ConfigurationError("LLM_BASE_URL must be a valid HTTP(S) API base URL.") from exc
        if (
            base_url.scheme not in ("http", "https")
            or not host
            or any(character.isspace() for character in self.LLM_BASE_URL)
            or base_url.username
            or base_url.password
            or base_url.query
            or base_url.fragment
        ):
            raise ConfigurationError(
                "LLM_BASE_URL must be an HTTP(S) API base URL without embedded credentials, query, or fragment."
            )

    def require_progrentis_credentials(self) -> tuple[str, str]:
        missing = [
            name
            for name, value in (
                ("PROGRENTIS_USERNAME", self.PROGRENTIS_USERNAME),
                ("PROGRENTIS_PASSWORD", self.PROGRENTIS_PASSWORD),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                f"Missing required Progrentis configuration: {', '.join(missing)}. "
                "Set these values in the project-root .env file."
            )
        return self.PROGRENTIS_USERNAME, self.PROGRENTIS_PASSWORD


def redact_sensitive_text(value: str) -> str:
    secrets = (
        settings.PROGRENTIS_PASSWORD,
        settings.PROGRENTIS_USERNAME,
        settings.LLM_API_KEY,
    )
    for secret in filter(None, secrets):
        variants = {
            secret,
            quote(secret, safe=""),
            quote_plus(secret),
            escape(secret),
        }
        for variant in variants:
            if variant:
                value = value.replace(variant, "[redacted]")
                value = value.replace(variant.lower(), "[redacted]")
    return value


def load_settings() -> Settings:
    """Load project-root environment variables and construct application settings."""
    load_dotenv(ROOT_DIR / ".env")
    return Settings()


settings = load_settings()
