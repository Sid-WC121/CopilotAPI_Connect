# Copilot-Ollama

Use GitHub Copilot Agent Mode with any OpenAI-compatible model provider — NVIDIA NIM, Cerebras, MiniMax, OpenRouter, OpenAI, and more.

## How It Works

Copilot's BYOK Ollama mode expects an Ollama server. This project runs two lightweight local services that bridge the gap:

```
VSCode Copilot  →  oai2ollama (:11434)  →  LiteLLM (:4000)  →  Provider API
```

- **oai2ollama** — speaks Ollama API to Copilot, forwards as OpenAI-compatible requests
- **LiteLLM** — routes each model to its configured provider and API key

Tool/function-calling is fully preserved, enabling Agent Mode.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) package manager
- VSCode with the GitHub Copilot extension
- An API key for at least one provider (see [Providers](#providers))

### Setup

1. Clone the repository.

```bash
git clone https://github.com/Sid-WC121/CopilotAPI_Connect.git
cd CopilotAPI_Connect
```

2. Create your `.env` file and add keys for the providers you use.

```env
# NVIDIA NIM — https://build.nvidia.com
NVIDIA_API_KEY="your_nvidia_api_key_here"

# Cerebras — https://cloud.cerebras.ai
# CEREBRAS_API_KEY="your_cerebras_api_key_here"
```

3. Start the proxy stack.

```bash
uv run python run.py
```

The launcher auto-selects free ports when defaults are busy:

- LiteLLM default: `4000`
- oai2ollama default: `11434`

If either port is occupied, the next free port is chosen and printed in the console.
Use the printed oai2ollama URL as your VSCode endpoint.

Alternative launchers: `run.cmd` (cmd), `./run.ps1` (PowerShell), `./run.sh` (bash).

4. Configure VSCode:
  - Set `github.copilot.chat.byok.ollamaEndpoint` to the URL printed by `run.py`
   - Open Manage Models → select Ollama

## Providers

Each model entry in `config.yaml` is independently configured with its own `api_key` and `api_base`. You can mix any number of providers.

| Tag suffix      | Provider                            | Needs `api_base`? | Key env var            |
| --------------- | ----------------------------------- | ------------------- | ---------------------- |
| `:nvidia`     | [NVIDIA NIM](https://build.nvidia.com) | Yes                 | `NVIDIA_API_KEY`     |
| `:cerebras`   | [Cerebras](https://cloud.cerebras.ai)  | No (built-in)       | `CEREBRAS_API_KEY`   |
| `:openrouter` | [OpenRouter](https://openrouter.ai)    | No (built-in)       | `OPENROUTER_API_KEY` |
| *(none)*      | OpenAI                              | No (built-in)       | `OPENAI_API_KEY`     |

The `:tag` suffix in `model_name` is what Copilot displays. It is stripped before routing so LiteLLM can match the entry correctly. Only `:latest` (auto-appended by the Ollama protocol) is fully stripped — custom tags are preserved end-to-end.

## Adding Models

Edit `config.yaml`:

```yaml
- model_name: my-model:provider        # shown in Copilot
  supports_function_calling: true
  supports_tool_choice: true
  supports_parallel_function_calling: true
  litellm_params:
    model: openai/org/model-id         # openai/ prefix = generic OpenAI-compatible
    api_base: https://api.provider.com/v1
    api_key: os.environ/MY_API_KEY
```
