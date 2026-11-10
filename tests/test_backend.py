import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import httpx
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from backend import agent, config, inference, main, tools
from backend.browser import Browser, BrowserError


class FakePage:
    def __init__(self):
        self.closed = False
        self.url = "https://example.test/"
        self.goto_status = 200
        self.goto_error = None

    def is_closed(self):
        return self.closed

    async def goto(self, url, **_kwargs):
        if self.goto_error:
            raise self.goto_error
        self.url = url
        return SimpleNamespace(status=self.goto_status)

    async def wait_for_load_state(self, *_args, **_kwargs):
        return None

    async def wait_for_timeout(self, _timeout):
        return None

    async def evaluate(self, _script, _arguments=None):
        return {
            "url": self.url,
            "title": "Example",
            "text": "Visible page text",
            "headings": [{"level": 1, "text": "Example"}],
            "links": [],
            "buttons": [],
            "inputs": [],
            "structure": {"main_regions": 1, "forms": 0, "tables": 0, "lists": 0},
        }


class ConfigurationTests(unittest.TestCase):
    def test_environment_file_is_loaded_from_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("HOST=127.0.0.9\nPORT=8123\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), patch.object(config, "ROOT_DIR", Path(directory)):
                loaded = config.load_settings()
            self.assertEqual(loaded.HOST, "127.0.0.9")
            self.assertEqual(loaded.PORT, 8123)

    def test_missing_llm_configuration_is_actionable(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "", "LLM_API_KEY": "", "LLM_MODEL": ""}, clear=False):
            loaded = config.Settings()
        with self.assertRaisesRegex(config.ConfigurationError, "LLM_BASE_URL.*LLM_API_KEY"):
            loaded.require_llm()

    def test_missing_progrentis_credentials_is_actionable(self):
        with patch.dict(os.environ, {"PROGRENTIS_USERNAME": "", "PROGRENTIS_PASSWORD": ""}, clear=False):
            loaded = config.Settings()
        with self.assertRaisesRegex(config.ConfigurationError, "PROGRENTIS_USERNAME.*PROGRENTIS_PASSWORD"):
            loaded.require_progrentis_credentials()


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_start_is_idempotent_and_shutdown_closes_resources(self):
        page = FakePage()
        context = SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
        chromium_instance = SimpleNamespace(
            new_context=AsyncMock(return_value=context), close=AsyncMock()
        )
        chromium = SimpleNamespace(launch=AsyncMock(return_value=chromium_instance))
        playwright = SimpleNamespace(chromium=chromium, stop=AsyncMock())
        with patch("backend.browser.async_playwright", return_value=SimpleNamespace(start=AsyncMock(return_value=playwright))):
            instance = Browser()
            await instance.start()
            await instance.start()
            self.assertTrue(instance.is_running)
            chromium.launch.assert_awaited_once()
            self.assertTrue(instance.is_headless)
            await instance.stop()
        context.close.assert_awaited_once()
        playwright.stop.assert_awaited_once()
        self.assertFalse(instance.is_running)
        self.assertIsNone(instance.is_headless)

    async def test_browser_can_switch_between_visible_and_headless_modes(self):
        pages = [FakePage(), FakePage()]
        contexts = [
            SimpleNamespace(new_page=AsyncMock(return_value=page), close=AsyncMock())
            for page in pages
        ]
        browsers = [
            SimpleNamespace(
                new_context=AsyncMock(return_value=context), close=AsyncMock()
            )
            for context in contexts
        ]
        chromium = SimpleNamespace(
            launch=AsyncMock(side_effect=browsers)
        )
        playwright = SimpleNamespace(chromium=chromium, stop=AsyncMock())
        with patch(
            "backend.browser.async_playwright",
            return_value=SimpleNamespace(start=AsyncMock(return_value=playwright)),
        ):
            instance = Browser()
            await instance.start(headless=True)
            await instance.start(headless=False)
        self.assertEqual(
            [call.kwargs["headless"] for call in chromium.launch.await_args_list],
            [True, False],
        )
        self.assertFalse(instance.is_headless)
        contexts[0].close.assert_awaited_once()
        self.assertTrue(instance.is_running)

    async def test_navigation_and_page_inspection_return_visible_observations(self):
        instance = Browser()
        instance.page = FakePage()
        result = await instance.navigate("https://example.test/path")
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(result["title"], "Example")
        self.assertEqual(result["source"], "browser_observation")
        self.assertEqual(result["url"], "https://example.test/path")
        self.assertIn("observed_at", result)

    async def test_navigation_timeout_has_clear_error(self):
        instance = Browser()
        instance.page = FakePage()
        instance.page.goto_error = PlaywrightTimeoutError("timed out")
        with self.assertRaisesRegex(BrowserError, "Navigation timed out"):
            await instance.navigate("https://example.test/")

    async def test_navigation_connection_reset_explains_network_failure(self):
        instance = Browser()
        instance.page = FakePage()
        instance.page.goto_error = PlaywrightError(
            "Page.goto: net::ERR_CONNECTION_RESET at https://example.test/"
        )
        with self.assertRaisesRegex(BrowserError, "ERR_CONNECTION_RESET.*firewall"):
            await instance.navigate("https://example.test/")

    async def test_inspection_redacts_account_values(self):
        instance = Browser()
        instance.page = FakePage()
        instance.page.evaluate = AsyncMock(return_value={
            "url": "https://example.test/",
            "title": "member@example.test",
            "text": "Welcome member@example.test",
            "headings": [], "links": [], "buttons": [], "inputs": [], "structure": {},
        })
        with patch.object(config.settings, "PROGRENTIS_USERNAME", "member@example.test"):
            result = await instance.inspect()
        self.assertNotIn("member@example.test", str(result))
        self.assertIn("[redacted]", result["text"])

    async def test_navigation_rejects_non_http_urls(self):
        instance = Browser()
        instance.page = FakePage()
        with self.assertRaisesRegex(ValueError, "HTTP and HTTPS"):
            await instance.navigate("file:///private.txt")

    async def test_progrentis_login_requires_origin_and_uses_local_credentials(self):
        page = FakePage()
        page.url = "https://progrentis.example/login"
        form_data = {
            "usernameIndex": 0,
            "passwordIndex": 1,
            "formOrigin": "https://progrentis.example",
            "submitSelector": "input",
            "submitIndex": 2,
        }
        page.evaluate = AsyncMock(side_effect=[
            form_data,
            {
                "url": "https://progrentis.example/dashboard",
                "title": "Dashboard",
                "text": "Welcome",
                "headings": [], "links": [], "buttons": [], "inputs": [], "structure": {},
            },
        ])
        filled = []
        clicked = []

        class Locator:
            def __init__(self, selector, index=None):
                self.selector = selector
                self.index = index

            def nth(self, _index):
                return Locator(self.selector, _index)

            async def fill(self, value):
                filled.append(value)

            async def click(self, **_kwargs):
                clicked.append((self.selector, self.index))
                return None

        page.locator = MagicMock(side_effect=lambda selector: Locator(selector))
        instance = Browser()
        instance.page = page
        with (
            patch("backend.browser.PROGRENTIS_HOST", "progrentis.example"),
            patch.object(config.settings, "PROGRENTIS_USERNAME", "local-user"),
            patch.object(config.settings, "PROGRENTIS_PASSWORD", "local-pass"),
        ):
            result = await instance.authenticate_progrentis("https://progrentis.example")
        self.assertEqual(filled, ["local-user", "local-pass"])
        self.assertEqual(clicked, [("input", 2)])
        self.assertEqual(result["status"], "form_submitted")
        self.assertNotIn("local-user", str(result))
        self.assertNotIn("local-pass", str(result))

    async def test_progrentis_login_refuses_unapproved_origin(self):
        page = FakePage()
        page.url = "https://other.example/login"
        page.evaluate = AsyncMock()
        instance = Browser()
        instance.page = page
        with (
            patch("backend.browser.PROGRENTIS_HOST", "progrentis.example"),
            patch.object(config.settings, "PROGRENTIS_USERNAME", "local-user"),
            patch.object(config.settings, "PROGRENTIS_PASSWORD", "local-pass"),
        ):
            with self.assertRaisesRegex(ValueError, "restricted to the confirmed Progrentis"):
                await instance.authenticate_progrentis("https://progrentis.example")
        page.evaluate.assert_not_awaited()


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def test_navigation_arguments_require_a_valid_absolute_url(self):
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            tools.validate_tool_arguments("navigate", {"url": "example.test"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            tools.validate_tool_arguments("navigate", {"url": "https://example.test", "extra": True})
        with self.assertRaisesRegex(ValueError, "does not accept"):
            tools.validate_tool_arguments("inspect_page", {"unexpected": True})

    async def test_tool_execution_calls_browser(self):
        with patch.object(tools.browser, "navigate", new=AsyncMock(return_value={"title": "Page"})) as navigate:
            result = await tools.execute_tool("navigate", {"url": "https://example.test"})
        self.assertEqual(result, {"title": "Page"})
        navigate.assert_awaited_once_with("https://example.test")


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_endpoint_emits_tokens_then_completion(self):
        state = SimpleNamespace(agent_task=None, last_error=None)
        request = SimpleNamespace(app=SimpleNamespace(state=state))

        async def fake_run_agent(_prompt, _history, on_token):
            await on_token("Hello")
            return agent.AgentResult("Hello!", [], 1, 0)

        with patch.object(main, "run_agent", side_effect=fake_run_agent):
            response = await main.agent_run_stream(
                main.AgentRequest(prompt="Say hello"),
                request,
            )
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]

        events = [json.loads(chunk) for chunk in chunks]
        self.assertEqual(events[0], {"type": "token", "content": "Hello"})
        self.assertEqual(events[1]["type"], "complete")
        self.assertEqual(events[1]["result"]["response"], "Hello!")
        self.assertIsNone(state.agent_task)

    async def test_agent_includes_recent_chat_history(self):
        chat = AsyncMock(return_value={"role": "assistant", "content": "I remember."})
        with (
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection=None)),
            patch.object(agent, "chat_completion", new=chat),
        ):
            result = await agent.run_agent(
                "What did I just ask?",
                [{"role": "user", "content": "Say hello."},
                 {"role": "assistant", "content": "Hello!"}],
            )
        self.assertEqual(result.response, "I remember.")
        messages = chat.await_args.args[0]
        self.assertEqual(
            [(message["role"], message["content"]) for message in messages[1:]],
            [
                ("user", "Say hello."),
                ("assistant", "Hello!"),
                ("user", "What did I just ask?"),
                ("assistant", "I remember."),
            ],
        )

    async def test_agent_refuses_to_click_and_does_not_call_model(self):
        chat = AsyncMock()
        with (
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection=None)),
            patch.object(agent, "chat_completion", new=chat),
        ):
            result = await agent.run_agent("Click it and return the page contents.")
        self.assertIn("Click it yourself in the visible Chromium window", result.response)
        chat.assert_not_awaited()

    async def test_multiple_tool_calls_return_observations_to_the_model(self):
        observations = [
            {"source": "browser_observation", "title": "One"},
            {"source": "browser_observation", "title": "Two"},
        ]
        tool_response = {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "inspect_page", "arguments": "{}"}},
                {"id": "call-2", "function": {"name": "inspect_page", "arguments": "{}"}},
            ],
        }
        final_response = {"role": "assistant", "content": "Observed two pages."}
        fake_browser = SimpleNamespace(is_running=True, last_inspection=None)
        with (
            patch.object(agent, "browser", fake_browser),
            patch.object(agent, "chat_completion", new=AsyncMock(side_effect=[tool_response, final_response])) as chat,
            patch.object(agent, "execute_tool", new=AsyncMock(side_effect=observations)) as execute,
        ):
            result = await agent.run_agent("Tell me about the weather")
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.observations, observations)
        self.assertEqual(result.iterations, 2)
        execute.assert_awaited()
        self.assertEqual(chat.await_count, 2)
        tool_messages = [
            message for message in chat.await_args_list[1].args[0]
            if message["role"] == "tool"
        ]
        self.assertEqual([message["tool_call_id"] for message in tool_messages], ["call-1", "call-2"])

    async def test_malformed_tool_arguments_are_returned_as_tool_errors(self):
        tool_response = {
            "role": "assistant",
            "tool_calls": [{
                "id": "bad-args",
                "function": {"name": "inspect_page", "arguments": "not-json"},
            }],
        }
        with (
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection=None)),
            patch.object(agent, "chat_completion", new=AsyncMock(side_effect=[
                tool_response, {"role": "assistant", "content": "The arguments were invalid."}
            ])),
            patch.object(agent, "execute_tool", new=AsyncMock()) as execute,
        ):
            result = await agent.run_agent("Inspect")
        self.assertEqual(result.response, "The arguments were invalid.")
        execute.assert_not_awaited()

    async def test_agent_iteration_limit_is_enforced(self):
        tool_response = {
            "role": "assistant",
            "tool_calls": [{
                "id": "loop",
                "function": {"name": "inspect_page", "arguments": "{}"},
            }],
        }
        with (
            patch.object(agent, "MAX_ITERATIONS", 2),
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection=None)),
            patch.object(agent, "chat_completion", new=AsyncMock(return_value=tool_response)) as chat,
            patch.object(agent, "execute_tool", new=AsyncMock(return_value={})),
        ):
            with self.assertRaisesRegex(RuntimeError, "iterations"):
                await agent.run_agent("Keep inspecting")
        self.assertEqual(chat.await_count, 2)

    async def test_page_request_collects_observation_before_model_without_tools(self):
        observation = {
            "source": "browser_observation",
            "url": "https://example.test/",
            "title": "Observed example",
            "text": "Actual visible text",
        }
        chat = AsyncMock(return_value={
            "role": "assistant", "content": "The page shows the observed example."
        })
        with (
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection={
                "source": "browser_observation", "title": "Different stale page"
            })),
            patch.object(agent, "execute_tool", new=AsyncMock(return_value=observation)) as execute,
            patch.object(agent, "chat_completion", new=chat),
        ):
            result = await agent.run_agent("Inspect this page and explain it")
        execute.assert_awaited_once_with("inspect_page", {})
        self.assertEqual(result.observations, [observation])
        self.assertEqual(result.response, "The page shows the observed example.")
        self.assertEqual(result.tool_calls, 0)
        self.assertEqual(len(chat.await_args.args), 1)
        self.assertIn("Actual visible text", chat.await_args.args[0][-1]["content"])
        self.assertNotIn("Different stale page", chat.await_args.args[0][-1]["content"])

    async def test_url_request_navigates_directly_and_passes_observation(self):
        observation = {
            "source": "browser_observation",
            "url": "https://example.test/",
            "title": "Example",
            "text": "Observed",
        }
        with (
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection=None)),
            patch.object(agent, "execute_tool", new=AsyncMock(return_value=observation)) as execute,
            patch.object(agent, "chat_completion", new=AsyncMock(return_value={
                "role": "assistant", "content": "I inspected the page."
            })) as chat,
        ):
            result = await agent.run_agent("Open https://example.test and describe the page")
        execute.assert_awaited_once_with("navigate", {"url": "https://example.test"})
        self.assertEqual(result.observations, [observation])
        self.assertEqual(chat.await_count, 1)

    def test_unsupported_tool_call_markup_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "non-OpenAI tool-call format"):
            agent._tool_calls({"content": "<tool_call>inspect_page</tool_call>"})

    async def test_credentials_are_redacted_before_and_after_model_call(self):
        with (
            patch.object(config.settings, "PROGRENTIS_USERNAME", "student@example.test"),
            patch.object(config.settings, "PROGRENTIS_PASSWORD", "private-password"),
            patch.object(agent, "browser", SimpleNamespace(is_running=True, last_inspection=None)),
            patch.object(agent, "chat_completion", new=AsyncMock(return_value={
                "role": "assistant", "content": "private-password"
            })) as chat,
        ):
            result = await agent.run_agent("Login as student@example.test with private-password")
        sent_prompt = chat.await_args.args[0][1]["content"]
        self.assertNotIn("student@example.test", sent_prompt)
        self.assertNotIn("private-password", sent_prompt)
        self.assertNotIn("private-password", result.response)
        self.assertIn("[redacted]", result.response)


class InferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_completion_yields_tokens_and_accumulates_tool_calls(self):
        class FakeStreamingResponse:
            def __init__(self, lines):
                self.lines = lines

            def raise_for_status(self):
                return None

            async def aiter_lines(self):
                for line in self.lines:
                    yield line

        class FakeStreamContext:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, *_args):
                return None

        class StreamingClient:
            def __init__(self):
                self.response = FakeStreamingResponse([
                    "data: " + json.dumps({"choices": [{
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }]}),
                    "data: " + json.dumps({"choices": [{
                        "delta": {"content": " there"},
                        "finish_reason": None,
                    }]}),
                    "data: " + json.dumps({"choices": [{
                        "delta": {"tool_calls": [{
                            "index": 0,
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "navigate", "arguments": "{\"url\":"},
                        }]},
                        "finish_reason": None,
                    }]}),
                    "data: " + json.dumps({"choices": [{
                        "delta": {"tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "\"https://example.test\"}"},
                        }]},
                        "finish_reason": "tool_calls",
                    }]}),
                    "data: [DONE]",
                ])
                self.stream = MagicMock(return_value=FakeStreamContext(self.response))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        client = StreamingClient()
        token_callback = AsyncMock()
        with (
            patch.object(config.settings, "LLM_BASE_URL", "https://model.example/v1"),
            patch.object(config.settings, "LLM_API_KEY", "local-api-key"),
            patch.object(config.settings, "LLM_MODEL", "nanbeige"),
            patch("backend.inference.httpx.AsyncClient", return_value=client),
        ):
            result = await inference.chat_completion(
                [{"role": "user", "content": "Hello"}],
                tools=[{"type": "function"}],
                stream=True,
                on_token=token_callback,
            )
        self.assertEqual(result["content"], "Hello there")
        self.assertEqual(result["tool_calls"][0]["id"], "call-1")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "navigate")
        self.assertEqual(
            result["tool_calls"][0]["function"]["arguments"],
            "{\"url\":\"https://example.test\"}",
        )
        self.assertEqual(
            [call.args[0] for call in token_callback.await_args_list],
            ["Hello", " there"],
        )
        self.assertTrue(client.stream.call_args.kwargs["json"]["stream"])

    async def test_openai_compatible_request_uses_bearer_auth_and_tool_schema(self):
        class SuccessfulClient:
            def __init__(self):
                self.post = AsyncMock(return_value=SimpleNamespace(
                    raise_for_status=MagicMock(),
                    json=MagicMock(return_value={
                        "choices": [{"message": {"role": "assistant", "content": "Ready"}}]
                    }),
                ))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        client = SuccessfulClient()
        with (
            patch.object(config.settings, "LLM_BASE_URL", "https://model.example/v1"),
            patch.object(config.settings, "LLM_API_KEY", "local-api-key"),
            patch.object(config.settings, "LLM_MODEL", "nanbeige"),
            patch("backend.inference.httpx.AsyncClient", return_value=client),
        ):
            result = await inference.chat_completion(
                [{"role": "user", "content": "Hello"}], tools=[{"type": "function"}]
            )
        self.assertEqual(result["content"], "Ready")
        args, kwargs = client.post.await_args
        self.assertEqual(args[0], "https://model.example/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer local-api-key")
        self.assertIn("tools", kwargs["json"])

    async def test_connection_failures_are_reported_without_request_secrets(self):
        class FailingClient:
            async def __aenter__(self):
                raise httpx.ConnectError("connection refused")

            async def __aexit__(self, *_args):
                return None

        with (
            patch.object(config.settings, "LLM_BASE_URL", "https://model.example/v1"),
            patch.object(config.settings, "LLM_API_KEY", "never-log-this"),
            patch.object(config.settings, "LLM_MODEL", "nanbeige"),
            patch("backend.inference.httpx.AsyncClient", return_value=FailingClient()),
        ):
            with self.assertRaisesRegex(RuntimeError, "Could not connect") as error:
                await inference.chat_completion([{"role": "user", "content": "hello"}])
        self.assertNotIn("never-log-this", str(error.exception))


if __name__ == "__main__":
    unittest.main()
