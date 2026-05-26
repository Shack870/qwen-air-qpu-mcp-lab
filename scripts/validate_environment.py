from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qpu_mcp_lab.config import llama_bin, llama_repo, model_path
from qpu_mcp_lab.quantum import credential_status


def main() -> None:
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(ROOT),
        "llama_bin": inspect_path(llama_bin()),
        "model_path": inspect_path(model_path()),
        "llama_repo": inspect_git_repo(llama_repo()),
        "macos": run_json(["sw_vers"]),
        "memory_bytes": run_text(["sysctl", "-n", "hw.memsize"]),
        "disk_root": run_json(["df", "-h", "/"]),
        "qiskit": import_status("qiskit"),
        "qiskit_ibm_runtime": import_status("qiskit_ibm_runtime"),
        "ibm_quantum_credentials": credential_status(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def inspect_path(path: Path) -> dict[str, object]:
    exists = path.exists()
    info: dict[str, object] = {"path": str(path), "exists": exists}
    if exists:
        info["size_bytes"] = path.stat().st_size
        info["executable"] = bool(path.stat().st_mode & 0o111)
    if exists and path.name.startswith("llama"):
        info["version"] = run_text([str(path), "--version"], timeout=10)
    return info


def inspect_git_repo(path: Path) -> dict[str, object]:
    info: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists() or not (path / ".git").exists():
        return info
    info["commit"] = run_text(["git", "-C", str(path), "rev-parse", "--short", "HEAD"])
    info["status_short"] = run_text(["git", "-C", str(path), "status", "--short"])
    return info


def import_status(module: str) -> dict[str, object]:
    if module not in sys.modules and shutil.which("python3") is None:
        return {"ok": False, "error": "python3 not found"}
    try:
        __import__(module)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True}


def run_json(cmd: list[str], timeout: int = 10) -> dict[str, object]:
    out = run(cmd, timeout=timeout)
    return {"command": cmd, **out}


def run_text(cmd: list[str], timeout: int = 10) -> str | None:
    out = run(cmd, timeout=timeout)
    if not out["ok"]:
        return None
    return out["stdout"].strip() or out["stderr"].strip()


def run(cmd: list[str], timeout: int = 10) -> dict[str, object]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return {"ok": False, "stdout": "", "stderr": "command not found", "returncode": 127}
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": "timeout", "returncode": None}
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


if __name__ == "__main__":
    main()
