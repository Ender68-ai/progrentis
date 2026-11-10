import asyncio
import logging
import re
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.async_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    async_playwright,
)

from backend.config import (
    PROGRENTIS_HOST,
    redact_sensitive_text,
    settings,
)

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 12000
ELEMENT_LIMIT = 60


class BrowserError(RuntimeError):
    """A safe, user-facing browser operation error."""


def _redact(value: object) -> object:
    if isinstance(value, str):
        return redact_sensitive_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def _safe_url(url: str) -> str:
    parts = urlsplit(url)
    safe_query = [
        (key, "[redacted]" if any(word in key.lower() for word in ("token", "key", "secret", "auth", "password")) else val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
    ]
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return redact_sensitive_text(
        urlunsplit((parts.scheme, host, parts.path, urlencode(safe_query), ""))
    )


class Browser:
    def __init__(self) -> None:
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._headless: bool | None = None
        self.last_inspection: dict | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self.page is not None and not self.page.is_closed()

    @property
    def is_headless(self) -> bool | None:
        return self._headless if self.is_running else None

    async def start(self, *, headless: bool = True) -> None:
        async with self._operation_lock:
            async with self._lifecycle_lock:
                if self.is_running and self._headless == headless:
                    return
                if self.playwright or self.browser or self.context:
                    await self._close_resources()

                try:
                    self.playwright = await async_playwright().start()
                    self.browser = await self.playwright.chromium.launch(headless=headless)
                    self.context = await self.browser.new_context()
                    self.page = await self.context.new_page()
                    self._headless = headless
                    self.last_inspection = None
                except Exception as exc:
                    try:
                        await self._close_resources()
                    except BrowserError as cleanup_error:
                        logger.error("Browser startup cleanup failed (%s)", type(cleanup_error).__name__)
                    if not isinstance(exc, PlaywrightError):
                        raise
                    logger.warning("Browser startup failed: %s", type(exc).__name__)
                    if not headless:
                        raise BrowserError(
                            "Could not open a visible Chromium window. Check that this app is running "
                            "in your desktop session and Playwright's Chromium is installed."
                        ) from exc
                    raise BrowserError(
                        "Could not start Chromium. Install the Playwright browser with "
                        "`python -m playwright install chromium` and check system dependencies."
                    ) from exc

    async def _close_resources(self) -> None:
        context, chromium, playwright = self.context, self.browser, self.playwright
        self.context = self.browser = self.playwright = self.page = None
        self._headless = None
        self.last_inspection = None
        errors: list[Exception] = []
        for resource in (context, chromium, playwright):
            if resource is None:
                continue
            try:
                await (resource.stop() if resource is playwright else resource.close())
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise BrowserError("One or more browser resources could not be shut down cleanly.") from errors[0]

    async def stop(self) -> None:
        async with self._operation_lock:
            async with self._lifecycle_lock:
                await self._close_resources()

    async def navigate(self, url: str) -> dict:
        try:
            parts = urlsplit(url)
            host = parts.hostname
            parts.port
        except ValueError as exc:
            raise ValueError("The browser URL must be a valid absolute HTTP or HTTPS URL.") from exc
        if (
            parts.scheme not in ("http", "https")
            or not host
            or any(character.isspace() for character in parts.netloc)
        ):
            raise ValueError("Only absolute HTTP and HTTPS URLs are allowed.")
        if parts.username or parts.password:
            raise ValueError("Browser URLs must not contain embedded credentials.")

        async with self._operation_lock:
            if not self.is_running:
                raise BrowserError("Browser is not running. Start the browser before navigating.")
            try:
                response = await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=2000)
                except PlaywrightTimeoutError:
                    pass
                await self.page.wait_for_timeout(300)
            except PlaywrightTimeoutError as exc:
                raise BrowserError(
                    f"Navigation timed out while loading {_safe_url(url)}. The page may still be partially available."
                ) from exc
            except PlaywrightError as exc:
                logger.warning("Navigation failed (%s)", type(exc).__name__)
                error_code = re.search(r"net::(ERR_[A-Z0-9_]+)", str(exc))
                if error_code:
                    code = error_code.group(1)
                    if code == "ERR_CONNECTION_RESET":
                        raise BrowserError(
                            f"Chromium could not connect to {_safe_url(url)} "
                            "(ERR_CONNECTION_RESET). Check your internet connection, "
                            "firewall, or network restrictions."
                        ) from exc
                    raise BrowserError(
                        f"Chromium could not load {_safe_url(url)} ({code}). "
                        "Check the address and network connection."
                    ) from exc
                raise BrowserError(
                    f"Could not navigate to {_safe_url(url)}. Check the address and network connection."
                ) from exc

            inspection = await self._inspect_page()
            inspection["http_status"] = response.status if response else None
            self.last_inspection = inspection
            return inspection

    async def inspect(self) -> dict:
        async with self._operation_lock:
            return await self._inspect_page()

    async def _inspect_page(self) -> dict:
        if not self.is_running:
            raise BrowserError("Browser is not running. Start the browser before inspecting a page.")

        try:
            await self.page.wait_for_timeout(150)
            result = await self.page.evaluate(
                """({textLimit, elementLimit}) => {
                    const visible = (el) => {
                        const style = getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return style.visibility !== "hidden" && style.display !== "none" &&
                               rect.width > 0 && rect.height > 0;
                    };
                    const text = (el) => (el.innerText || el.textContent || "").trim();
                    const elements = (selector, map) =>
                        [...document.querySelectorAll(selector)].filter(visible)
                            .slice(0, elementLimit).map(map);
                    return {
                        url: location.href,
                        title: document.title,
                        text: (document.body?.innerText || "").slice(0, textLimit),
                        headings: elements("h1,h2,h3,h4,h5,h6", el => ({
                            level: Number(el.tagName.slice(1)), text: text(el).slice(0, 500)
                        })),
                        links: elements("a[href]", el => ({
                            text: text(el).slice(0, 300), href: el.href
                        })),
                        buttons: elements("button,[role=button],input[type=button],input[type=submit]", el => ({
                            text: (el.innerText || el.value || el.getAttribute("aria-label") || "").trim().slice(0, 300),
                            disabled: Boolean(el.disabled),
                            type: el.getAttribute("type") || el.getAttribute("role") || "button"
                        })),
                        inputs: elements("input,textarea,select", el => ({
                            type: el.type || el.tagName.toLowerCase(),
                            placeholder: el.getAttribute("placeholder") || "",
                            name: el.getAttribute("name") || "",
                            label: el.labels ? [...el.labels].map(label => text(label)).join(" ").slice(0, 300) : "",
                            aria_label: el.getAttribute("aria-label") || "",
                            autocomplete: el.getAttribute("autocomplete") || "",
                            required: Boolean(el.required),
                            disabled: Boolean(el.disabled)
                        })),
                        structure: {
                            main_regions: document.querySelectorAll("main,[role=main]").length,
                            forms: document.querySelectorAll("form").length,
                            tables: document.querySelectorAll("table").length,
                            lists: document.querySelectorAll("ul,ol,[role=list]").length
                        }
                    };
                }""",
                {"textLimit": MAX_TEXT_LENGTH, "elementLimit": ELEMENT_LIMIT},
            )
        except PlaywrightError as exc:
            logger.warning("Page inspection failed (%s)", type(exc).__name__)
            raise BrowserError("Could not inspect the current page. It may have navigated or closed.") from exc

        result["url"] = _safe_url(result["url"])
        result["source"] = "browser_observation"
        result["observed_at"] = datetime.now(timezone.utc).isoformat()
        self.last_inspection = _redact(result)
        return self.last_inspection

    async def authenticate_progrentis(self, authorized_origin: str) -> dict:
        async with self._operation_lock:
            return await self._authenticate_progrentis(authorized_origin)

    async def _authenticate_progrentis(self, authorized_origin: str) -> dict:
        if not self.is_running:
            raise BrowserError("Browser is not running. Open the Progrentis login page first.")

        username, password = settings.require_progrentis_credentials()
        try:
            current = urlsplit(self.page.url)
            approved = urlsplit(authorized_origin)
            current_port = current.port
            approved_port = approved.port
        except ValueError as exc:
            raise ValueError("Confirm the exact Progrentis HTTPS origin currently shown in the browser.") from exc
        current_origin = f"{current.scheme}://{current.netloc}"
        approved_origin = f"{approved.scheme}://{approved.netloc}"
        if (
            current.scheme != "https"
            or current.hostname != PROGRENTIS_HOST
            or current_port not in (None, 443)
            or approved_origin != current_origin
            or approved.path not in ("", "/")
            or approved.query
            or approved.fragment
            or approved.username
            or approved.password
            or approved.hostname != PROGRENTIS_HOST
            or approved.scheme != "https"
            or approved_port not in (None, 443)
        ):
            raise ValueError("Sign-in is restricted to the confirmed Progrentis HTTPS origin.")

        stage = "examining the sign-in form"
        try:
            form = await self.page.evaluate(
                """() => {
                    const visible = (el) => {
                        const style = getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return style.visibility !== "hidden" && style.display !== "none" &&
                               rect.width > 0 && rect.height > 0;
                    };
                    const text = (el) => [
                        el.name, el.id, el.placeholder, el.getAttribute("aria-label"),
                        el.autocomplete, ...(el.labels || []).map(label => label.innerText)
                    ].filter(Boolean).join(" ").toLowerCase();
                    const passwordFields = [...document.querySelectorAll('input[type="password"]')]
                        .filter(visible);
                    const captcha = [...document.querySelectorAll("iframe,[data-sitekey]")]
                        .filter(visible).some(el => /captcha|challenge/i.test(
                            `${el.src || ""} ${el.title || ""} ${el.getAttribute("data-sitekey") || ""}`
                        )) || /captcha|verify you are human/i.test(document.body?.innerText || "");
                    if (captcha) return {error: "A CAPTCHA or human-verification challenge was detected."};
                    if (passwordFields.length !== 1)
                        return {error: "Expected exactly one visible password field; login fields are ambiguous."};
                    const password = passwordFields[0];
                    const owner = password.form;
                    if (!owner) return {error: "The visible password field is not inside a form."};
                    const inputs = [...owner.querySelectorAll('input:not([type="password"])')]
                        .filter(el => visible(el) && !["hidden", "submit", "button", "checkbox", "radio"]
                            .includes((el.type || "").toLowerCase()));
                    if (!inputs.length) return {error: "No visible username or email field was found."};
                    const named = inputs.filter(el => /user|usuario|correo|email|login|account|identifier/.test(text(el)));
                    const candidates = named.length === 1 ? named : inputs;
                    if (candidates.length !== 1)
                        return {error: "Could not uniquely identify the username field from the login form."};
                    const submitters = [...owner.querySelectorAll(
                        'input[type="submit"],button[type="submit"],button:not([type])'
                    )].filter(visible);
                    const loginActions = submitters.filter(el =>
                        /ingresar|iniciar[ ]*sesi[oó]n|entrar|acceder|sign[ ]*in|log[ ]*in/i.test(
                            el.value || el.innerText || el.getAttribute("aria-label") || ""
                        )
                    );
                    if (loginActions.length !== 1)
                        return {error: "Could not uniquely identify the sign-in action; credentials were not entered."};
                    const allInputs = [...document.querySelectorAll("input")];
                    const allButtons = [...document.querySelectorAll("button")];
                    const submitter = loginActions[0];
                    return {
                        usernameIndex: allInputs.indexOf(candidates[0]),
                        passwordIndex: allInputs.indexOf(password),
                        formOrigin: new URL(owner.action || location.href, location.href).origin,
                        submitSelector: submitter.tagName === "INPUT" ? "input" : "button",
                        submitIndex: submitter.tagName === "INPUT"
                            ? allInputs.indexOf(submitter) : allButtons.indexOf(submitter)
                    };
                }"""
            )
            if form.get("error"):
                raise BrowserError(form["error"])
            if form["formOrigin"] != current_origin:
                raise BrowserError(
                    "The login form submits to a different website origin; credentials were not entered."
                )
            stage = "entering the username"
            await self.page.locator("input").nth(form["usernameIndex"]).fill(username)
            stage = "entering the password"
            await self.page.locator("input").nth(form["passwordIndex"]).fill(password)
            stage = "submitting the sign-in form"
            await self.page.locator(form["submitSelector"]).nth(form["submitIndex"]).click(timeout=5000)
            stage = "waiting for the sign-in page to respond"
            try:
                await self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            except PlaywrightTimeoutError:
                await self.page.wait_for_timeout(750)
            stage = "checking the page after submission"
            observation = await self._inspect_page()
        except (BrowserError, ValueError):
            raise
        except PlaywrightError as exc:
            logger.warning(
                "Progrentis login flow failed during %s (%s)",
                stage,
                type(exc).__name__,
            )
            raise BrowserError(
                f"Could not complete the login form while {stage}. "
                "Inspect the page and verify its fields manually."
            ) from exc

        return {
            "status": "form_submitted",
            "message": "The login form was submitted. Authentication success has not been assumed.",
            "observation": observation,
        }

    async def get_page(self):
        if not self.is_running:
            raise BrowserError("Browser is not running.")
        return self.page


browser = Browser()
