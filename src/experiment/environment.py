r"""Capture the runtime environment for an experiment: software versions, GPU
info, git commit, and a pip freeze. Everything is best-effort -- a missing tool
records "unavailable" rather than crashing a long run.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict

import torch


def gpu_info() -> Dict[str, Any]:
    """CUDA availability + device details. Explicit about CPU fallback."""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version()
                          if torch.backends.cudnn.is_available() else None),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "device_name": props.name,
            "device_count": torch.cuda.device_count(),
            "total_memory_gb": round(props.total_memory / 1e9, 2),
        })
    else:
        info["note"] = ("CUDA not available -- running on CPU. The GCP training "
                        "configuration expects a GPU; CPU is intended only for "
                        "local smoke tests.")
    return info


def software_versions() -> Dict[str, str]:
    """Versions of the key scientific dependencies (best-effort)."""
    versions = {"python": sys.version.split()[0], "platform": platform.platform(),
                "torch": torch.__version__}
    for mod in ("numpy", "scipy", "nibabel", "torchio", "cv2",
                "matplotlib", "tensorboard", "yaml"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "unavailable"
    return versions


def _run(cmd: list) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return ""


def git_info() -> Dict[str, Any]:
    """Commit hash, branch and dirty-state so results are traceable to source."""
    commit = _run(["git", "rev-parse", "HEAD"])
    if not commit:
        return {"available": False}
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}


def pip_freeze() -> str:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    return out or "pip freeze unavailable"


def write_environment(ws) -> Dict[str, Any]:
    """Write config/environment.txt, config/pip_freeze.txt, config/git_commit.txt
    into the experiment workspace and return a summary dict for the manifest."""
    versions = software_versions()
    gpu = gpu_info()
    git = git_info()

    with open(ws.path("config", "environment.txt"), "w", encoding="utf-8") as fh:
        fh.write("== Software versions ==\n")
        for k, v in versions.items():
            fh.write(f"{k}: {v}\n")
        fh.write("\n== GPU / CUDA ==\n")
        for k, v in gpu.items():
            fh.write(f"{k}: {v}\n")

    with open(ws.path("config", "pip_freeze.txt"), "w", encoding="utf-8") as fh:
        fh.write(pip_freeze() + "\n")

    with open(ws.path("config", "git_commit.txt"), "w", encoding="utf-8") as fh:
        if git.get("available"):
            fh.write(f"commit: {git['commit']}\nbranch: {git['branch']}\n"
                     f"dirty: {git['dirty']}\n")
        else:
            fh.write("git: repository information unavailable\n")

    return {"software": versions, "gpu": gpu, "git": git}
