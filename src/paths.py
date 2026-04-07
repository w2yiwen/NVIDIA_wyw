"""Resolve workspace root and common directories for local vs Kaggle."""

from __future__ import annotations

import os
from pathlib import Path


def workspace_root() -> Path:
    env = os.environ.get("NEMOTRON_WORKSPACE")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (workspace_root() / path).resolve()


def apply_disk_cache_env(weights_disk_root: str | Path) -> None:
    """
    Put Hugging Face + Kaggle Hub downloads on a data volume (e.g. AutoDL /root/autodl-tmp).
    Call this before any hub download or from_pretrained that may fetch files.
    """
    root = Path(weights_disk_root).expanduser().resolve()
    hf_home = root / "huggingface"
    kaggle_hub = root / "kagglehub"
    tmp = root / "tmp"
    for p in (hf_home, kaggle_hub, tmp):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["KAGGLEHUB_CACHE"] = str(kaggle_hub)
    os.environ.setdefault("TMPDIR", str(tmp))
    print(f"[cache] HF_HOME={hf_home}")
    print(f"[cache] KAGGLEHUB_CACHE={kaggle_hub}")
    print(f"[cache] TMPDIR={os.environ['TMPDIR']}")
