from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 5
    for process in processes:
        if process.poll() is not None:
            continue
        remaining = max(0, deadline - time.time())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    load_env_file(project_root / ".env")

    # Default config.yaml targets NVIDIA's endpoint. For that setup, a NVIDIA key is required.
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    litellm_key = os.getenv("LITELLM_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    # If user only sets NVIDIA_API_KEY, mirror it into LITELLM_API_KEY for consistency.
    if nvidia_key and not litellm_key:
        os.environ["LITELLM_API_KEY"] = nvidia_key
        litellm_key = nvidia_key

    if not nvidia_key:
        if openrouter_key and not litellm_key:
            print("OPENROUTER_API_KEY is set, but default config.yaml uses NVIDIA endpoint.")
            print("Set NVIDIA_API_KEY, or change config.yaml model/api_base to an OpenRouter model.")
        elif not litellm_key:
            print("No API key found. Create .env from .env.example and set NVIDIA_API_KEY.")
        else:
            print("LITELLM_API_KEY is set, but NVIDIA_API_KEY is missing for default NVIDIA config.")
            print("Set NVIDIA_API_KEY, or update config.yaml to use a provider matching your key.")
        return 1

    uv_path = shutil.which("uv")
    if not uv_path:
        print("uv is not installed or not on PATH.")
        return 1

    processes: list[subprocess.Popen[bytes]] = []

    def _handle_signal(_signum: int, _frame: object) -> None:
        print("\nStopping background processes...")
        terminate_processes(processes)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    litellm = subprocess.Popen([uv_path, "run", "litellm", "--config", "config.yaml"])
    processes.append(litellm)
    print(f"Started litellm with PID {litellm.pid}")

    oai2ollama = subprocess.Popen([uv_path, "run", "oai2ollama", "--api-key", "any", "--base-url", "http://localhost:4000"])
    processes.append(oai2ollama)
    print(f"Started oai2ollama with PID {oai2ollama.pid}")

    try:
        while True:
            if litellm.poll() is not None:
                print(f"litellm exited with code {litellm.returncode}")
                return litellm.returncode or 0
            if oai2ollama.poll() is not None:
                print(f"oai2ollama exited with code {oai2ollama.returncode}")
                return oai2ollama.returncode or 0
            time.sleep(1)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
