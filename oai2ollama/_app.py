import logging
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from .config import env

logger = logging.getLogger("oai2ollama")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(">>> %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("<<< %s %s %s", request.method, request.url.path, response.status_code)
    return response

_OLLAMA_DETAILS = {
    "parent_model": "",
    "format": "gguf",
    "family": "llm",
    "families": ["llm"],
    "parameter_size": "unknown",
    "quantization_level": "unknown",
}


def _ollama_name(model_id: str) -> str:
    """Ensure model name has an Ollama-style :tag suffix."""
    return model_id if ":" in model_id else f"{model_id}:latest"


def _litellm_name(model_id: str) -> str:
    """Strip auto-appended :latest suffix only; preserve custom tags (e.g. :nvidia, :cerebras)
    so they continue to match the model_name entries in config.yaml."""
    if model_id.endswith(":latest"):
        return model_id[: -len(":latest")]
    return model_id


def _new_client():
    from httpx import AsyncClient

    return AsyncClient(base_url=str(env.base_url), headers={"Authorization": f"Bearer {env.api_key}"}, timeout=60, http2=True)


def _upstream_error_response(status_code: int, content: bytes):
    try:
        return JSONResponse(status_code=status_code, content=json.loads(content.decode("utf-8")))
    except Exception:
        return PlainTextResponse(content.decode("utf-8", errors="replace"), status_code=status_code)


@app.get("/")
async def root():
    return PlainTextResponse("Ollama is running")


@app.get("/api/ps")
async def ps():
    # Return an empty running-models list; Copilot uses this to verify server is alive.
    return {"models": []}


@app.get("/api/tags")
async def models():
    async with _new_client() as client:
        res = await client.get("/models")
        res.raise_for_status()
    entries = []
    for item in res.json()["data"]:
        name = _ollama_name(item["id"])
        entries.append({
            "name": name,
            "model": name,
            "modified_at": "2025-01-01T00:00:00Z",
            "size": 0,
            "digest": "0" * 64,
            "details": _OLLAMA_DETAILS,
        })
    return {"models": entries}


@app.get("/v1/models")
async def v1_models():
    async with _new_client() as client:
        res = await client.get("/models")
        res.raise_for_status()
    items = []
    for item in res.json()["data"]:
        items.append({"id": item["id"], "object": "model", "owned_by": "ollama"})
    return {"object": "list", "data": items}


@app.get("/api/version")
async def version():
    return {"version": "0.6.5"}


@app.post("/api/show")
async def show_model(request: Request):
    body = await request.json()
    raw_name = body.get("name", body.get("model", "unknown"))
    name = _ollama_name(raw_name)
    return {
        "name": name,
        "model": name,
        "modified_at": "2025-01-01T00:00:00Z",
        "size": 0,
        "digest": "0" * 64,
        "details": _OLLAMA_DETAILS,
        "model_info": {"general.architecture": "CausalLM"},
        "capabilities": ["chat", "tools", "stop", "reasoning"],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()

    if "model" in data:
        data = {**data, "model": _litellm_name(data["model"])}

    if data.get("stream", False):
        async with _new_client() as client:
            # Use a non-stream upstream call and adapt it to SSE to avoid lifecycle
            # issues from holding upstream network streams open across response boundaries.
            upstream_payload = {**data, "stream": False}
            res = await client.post("/chat/completions", json=upstream_payload)
            if res.is_error:
                return _upstream_error_response(res.status_code, res.content)

            completion = res.json()
            choices = completion.get("choices")
            if not isinstance(choices, list) or not choices:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {
                            "message": "Upstream completion response did not include choices",
                            "type": "bad_upstream_response",
                            "code": "502",
                        }
                    },
                )

            choice0 = choices[0]
            message = choice0.get("message") if isinstance(choice0, dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
            completion_id = completion.get("id", "chatcmpl-oai2ollama")
            created = completion.get("created", int(time.time()))
            model = completion.get("model", data.get("model", "unknown"))
            finish_reason = choice0.get("finish_reason", "stop") if isinstance(choice0, dict) else "stop"

            async def stream():
                role_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8")

                if tool_calls:
                    tool_calls_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(tool_calls_chunk)}\n\n".encode("utf-8")

                if content:
                    content_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(content_chunk)}\n\n".encode("utf-8")

                stop_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

    else:
        async with _new_client() as client:
            res = await client.post("/chat/completions", json=data)
            if res.is_error:
                return _upstream_error_response(res.status_code, res.content)
            return res.json()
