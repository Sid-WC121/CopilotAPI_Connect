from ._app import app


def start():
    import os

    import uvicorn

    host = os.getenv("OAI2OLLAMA_HOST", "localhost")
    port = int(os.getenv("OAI2OLLAMA_PORT", "11434"))
    uvicorn.run(app, host=host, port=port)


__all__ = ["app", "start"]
