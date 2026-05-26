from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DB_PATH = DATA_DIR / "qpu_lab.sqlite"
CONFIG_PATH = ROOT / "config.json"

DEFAULT_LLAMA_BIN = Path("~/src/ik_llama.cpp/build-air-iqk-lean/bin/llama-cli").expanduser()
DEFAULT_MODEL_PATH = Path(
    "~/qwen-air-tests/models/byteshape-qwen3-30b-a3b-2507/"
    "Qwen3-30B-A3B-Instruct-2507-Q3_K_S-2.66bpw.gguf"
).expanduser()
DEFAULT_LLAMA_REPO = Path("~/src/ik_llama.cpp").expanduser()
DEFAULT_SAFE_MEMORY_GB = 6.5

PROMPTS: dict[str, str] = {
    "mars_capital": "What is the capital of Mars? Keep the answer concise.",
    "mars_fact_list": (
        "<|im_start|>user\n"
        "List concise facts about Mars as comma-separated phrases."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "hello_python": "Write one Python print statement.",
    "prime_python": "Write a compact Python function that checks whether a number is prime.",
    "binary_search": "Write a compact Python binary search function.",
    "qwen_short": "Explain in two sentences what a mixture-of-experts model is.",
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_PATH} must contain a JSON object")
    return data


def setting(name: str, default: Any) -> Any:
    env_name = f"QPU_MCP_LAB_{name.upper()}"
    return os.environ.get(env_name, load_config().get(name, default))


def llama_bin() -> Path:
    return Path(setting("llama_bin", str(DEFAULT_LLAMA_BIN))).expanduser()


def model_path() -> Path:
    return Path(setting("model_path", str(DEFAULT_MODEL_PATH))).expanduser()


def llama_repo() -> Path:
    return Path(setting("llama_repo", str(DEFAULT_LLAMA_REPO))).expanduser()


def safe_memory_gb() -> float:
    return float(setting("safe_memory_gb", DEFAULT_SAFE_MEMORY_GB))
