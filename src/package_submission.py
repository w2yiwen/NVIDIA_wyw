#!/usr/bin/env python3
"""Build submission.zip with LoRA adapter files (competition format)."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import yaml

from src.paths import resolve_path


def _load_required(config_path: Path | None) -> list[str]:
    if config_path and config_path.is_file():
        with config_path.open() as f:
            cfg = yaml.safe_load(f)
        return list(cfg.get("submission", {}).get("required_in_zip", []))
    return ["adapter_config.json", "adapter_model.safetensors"]


def package_adapter(
    adapter_dir: Path,
    zip_path: Path,
    required: list[str],
) -> None:
    adapter_dir = adapter_dir.resolve()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")

    files = {p.name for p in adapter_dir.iterdir() if p.is_file()}
    missing = [r for r in required if r not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {adapter_dir}: {missing}. Found: {sorted(files)}"
        )

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(files):
            fpath = adapter_dir / name
            zf.write(fpath, arcname=name)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created {zip_path} ({size_mb:.1f} MB)")
    with zipfile.ZipFile(zip_path, "r") as zf:
        print("Zip contents:", zf.namelist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--zip_path", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    cfg = {}
    if cfg_path.is_file():
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}

    adapter_dir = resolve_path(
        args.adapter_dir or cfg.get("training", {}).get("output_dir", "outputs/adapter")
    )
    zip_path = resolve_path(
        args.zip_path or cfg.get("submission", {}).get("zip_path", "outputs/submission.zip")
    )
    required = _load_required(cfg_path)

    package_adapter(adapter_dir, zip_path, required)
    print("submission.zip is ready for upload.")


if __name__ == "__main__":
    main()
