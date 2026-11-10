# Progrentis Agent

A local FastAPI application for read-only browser inspection and tutoring support. It combines Playwright/Chromium with an OpenAI-compatible llama.cpp endpoint. The web interface and backend run on your computer; the configured prompt and observed page content are sent to the configured LLM service for interpretation.

The agent does not submit assignments, change account settings, bypass CAPTCHAs, or claim observations it did not make. Progrentis sign-in is a separate, explicit browser action and uses credentials loaded locally.

## Requirements

- Python 3.10 or later
- Chromium dependencies supported by Playwright
- A reachable OpenAI-compatible llama.cpp server and its API key
- Optional: a Progrentis test account for authorized inspection

## Install

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install chromium
Copy-Item .env.example .env
```

If PowerShell prevents environment activation, either allow local scripts for your user or invoke `.venv\Scripts\python.exe` directly for the commands below.

### Fedora

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m playwright install --with-deps chromium
cp .env.example .env
```

Fedora may require the Playwright system libraries. `playwright install --with-deps chromium` installs supported Linux dependencies; if package installation is restricted, use the Fedora packages reported by Playwright's error output.

## Configuration

Edit the project-root `.env` file. `.env` is ignored by Git; do not commit it or share its contents.

| Variable | Required | Description |
|---|---:|---|
| `HOST` | No | Bind address; keep `127.0.0.1` for local-only use. |
| `PORT` | No | Local backend port (default `8000`). |
| `LOG_LEVEL` | No | Python log level (default `INFO`). Prompts and credentials are not logged. |
| `LLM_BASE_URL` | Yes | llama.cpp OpenAI-compatible API base, usually `https://<tunnel-host>/v1`. |
| `LLM_API_KEY` | Yes | API key configured on the inference server. |
| `LLM_MODEL` | Yes | Model identifier accepted by the server; default `nanbeige-4.3-33b`. |
| `LLM_TIMEOUT` | No | Per-request timeout in seconds (default `300`). |
| `PROGRENTIS_USERNAME` | For sign-in | Account username; kept local and removed from prompt/output text. |
| `PROGRENTIS_PASSWORD` | For sign-in | Account password; kept local and never included in browser inspection output. |

Configuration is loaded with `python-dotenv` from the repository root. Start the server again after changing `.env`. Missing LLM settings or account credentials produce a specific configuration error when that feature is used.

## Run the entire agent in Kaggle (no inference tunnel)

For a single-session setup, upload [progrentis_agent_kaggle.ipynb](./progrentis_agent_kaggle.ipynb) to Kaggle and attach the same `nanbeige-4-3-33b-gguf` and `llama-cpp-bin` datasets used by the Nanbeige server notebook. Enable a GPU and Internet access, then run the cells in order. The notebook starts llama.cpp on loopback and runs the existing agent and Playwright browser in that Kaggle runtime; its notebook-widget UI does not expose a public server or use Cloudflare. The browser therefore accesses websites from Kaggle, not from your laptop. Push the latest project source to GitHub `main` or attach it as a Kaggle dataset before starting if it is not already available there.

This removes Cloudflare from the inference path (and so avoids Cloudflare's 524 origin timeout). It does not prevent model/GPU startup failures or make Kaggle's own session and resource limits unlimited.

## Kaggle llama.cpp inference

1. Attach the Nanbeige 4.3 33B Q4_K_M GGUF to a Kaggle notebook with an enabled GPU.
2. Install or build `llama.cpp` in that notebook, then start `llama-server` with the attached model, a supported chat template, and OpenAI-compatible tool calling enabled. A typical command is:

   ```bash
   ./llama.cpp/build/bin/llama-server \
     --model "/kaggle/input/<your-model-dataset>/<model-file>.gguf" \
     --host 0.0.0.0 --port 8001 \
     --api-key "$LLM_API_KEY" \
     --jinja --ctx-size 8192 --n-gpu-layers <layers-for-your-GPU>
   ```

   Set `LLM_API_KEY` in Kaggle Secrets and export it into the notebook process without printing it. Adjust the binary path, model file and GPU layer count to match the notebook and available GPU memory.
3. Keep the inference process running. The local backend sends chat completions to `/v1/chat/completions` using a Bearer token and requests OpenAI-compatible `tools` when using browser tools.
4. If the selected model/chat template cannot emit OpenAI-compatible `tool_calls`, the agent reports an unsupported tool-call format instead of treating generated text as an executed browser action.

The model server and its tool-call template must be tested together; model aliases do not guarantee tool support.

## Cloudflare Quick Tunnel

Run this in a second Kaggle notebook cell while `llama-server` is listening:

```bash
cloudflared tunnel --url http://127.0.0.1:8001
```

Copy the `https://…trycloudflare.com` address printed by `cloudflared` and set `LLM_BASE_URL` to that address plus `/v1`, for example `https://<temporary-host>.trycloudflare.com/v1`. Do not include `/chat/completions`; the backend adds that path. Quick Tunnel hostnames can change when restarted. Use the same API key in the server and local `.env`. Treat the tunnel URL and key as private, and stop the tunnel when finished.

## Start the application

From the project root, with the virtual environment active:

```bash
python -m backend.main
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app). The backend JSON health endpoint remains at `/health`, and runtime status is available at `/status`. In the browser panel, select **Visible window** to open Chromium on your desktop and interact with it directly; use **Inspect** after a page change to share a fresh observation with the assistant. Select **Headless** to use the original background mode. Agent runs that start their own browser use headless mode. The chat includes up to four recent user/assistant turns for context; that context is reset when the page is reloaded.

Use **Progrentis sign-in** to open the supplied `https://prod.progrentis.com/signin/tabid/1543/Default.aspx` page, or enter another URL explicitly. Inspection includes visible text, headings, links, buttons, input metadata and high-level structure. It does not collect input values. The `Sign in to current page` action is separate from the model and requires explicit confirmation. Sign-in is restricted to the confirmed `https://prod.progrentis.com` origin. It discovers a single visible password field, a matching username field and the labeled sign-in action at runtime, refuses ambiguous forms and human-verification challenges, checks that the form submits to the same origin, and reports only that the form was submitted—not that authentication succeeded.

## Testing

Run the automated tests with the standard library:

```bash
python -m unittest discover -s tests -v
```

Tests use mocks for Playwright and the LLM and do not need a Kaggle session or real Progrentis account.

## API overview

- `POST /browser/start`, `/browser/stop`
- `POST /browser/navigate` with `{"url":"https://..."}` (HTTP(S) only)
- `POST /browser/inspect`
- `POST /agent/run` with `{"prompt":"...","history":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`; up to four recent turns are accepted and the response separates interpretation from browser observations.
- `POST /agent/run/stream` accepts the same payload and streams newline-delimited JSON token, completion, and error events. The GUI uses this endpoint to display generated text as it arrives.
- `POST /agent/stop` cancels the active run.
- `POST /progrentis/login` requires the confirmed current origin; credentials are read from `.env`.
- `POST /progrentis/open-login` opens the supplied Progrentis sign-in page for review.
- `GET /status`, `/health`

The agent caps model iterations and browser tool calls per run. It does not use a model-returned tool call unless the response is valid OpenAI-compatible tool-call data.

## Troubleshooting

- **Missing LLM settings:** Fill `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in the root `.env`; verify the tunnel is active and the base URL ends in `/v1`.
- **LLM timeout/connection refused:** Confirm the Kaggle notebook is running, the Quick Tunnel points to the llama.cpp port, and local network access is available.
- **HTTP 524 during a streamed response:** Streaming displays output incrementally once the model starts sending tokens, but cannot override a timeout imposed by Cloudflare or the inference provider before the first token.
- **HTTP 401/403 from inference:** Check that the server API key and local `LLM_API_KEY` match. Never print the key to troubleshoot.
- **Model responds without tool calls:** Confirm llama.cpp is serving an OpenAI-compatible chat template with tool calling enabled (for example, `--jinja`). Some model/template combinations do not support tools.
- **Chromium launch failure:** Run `python -m playwright install chromium` on Windows or `python -m playwright install --with-deps chromium` on Fedora.
- **Navigation timeout:** Check the address and connectivity. A timeout can leave a partially loaded page available to inspect.
- **Login fields are ambiguous or a challenge is detected:** The agent deliberately stops rather than guessing fields or bypassing a CAPTCHA/access control. Sign in manually or inspect the page yourself.
- **Credentials missing:** Add `PROGRENTIS_USERNAME` and `PROGRENTIS_PASSWORD` to `.env`. Do not paste credentials into chat prompts.
