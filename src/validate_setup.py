#!/usr/bin/env python3
"""Smoke checks: train.csv schema, imports, optional packaging dry-run."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

from src.data_pipeline import attach_training_text, load_train_dataframe, rows_to_hf_dataset
from src.paths import resolve_path
from src.package_submission import package_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip_torch", action="store_true", help="Only validate CSV + zip logic")
    args = parser.parse_args()

    cfg_path = resolve_path(args.config)
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    train_csv = resolve_path(cfg["data"]["train_csv"])
    print(f"Loading {train_csv} ...")
    df = load_train_dataframe(train_csv, subsample_size=5)
    print(f"  rows={len(df)}, columns={df.columns}")

    if args.skip_torch:
        print("skip_torch: done.")
        return

    import torch
    from transformers import AutoTokenizer

    model_path = cfg["model"].get("local_smoke_model")
    if not model_path:
        print("No model.local_smoke_model in config; skipping tokenizer map test.")
        print("Add a small local model path to configs/default.yaml to test full map().")
        return

    mp = resolve_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf = rows_to_hf_dataset(df.head(2))
    mapped = attach_training_text(hf, tokenizer)
    print("Sample text field:", mapped[0]["text"][:200], "...")

    # Packaging dry-run with minimal fake adapter files
    req = cfg.get("submission", {}).get("required_in_zip", ["adapter_config.json", "adapter_model.safetensors"])
    with tempfile.TemporaryDirectory() as td:
        ad = Path(td)
        (ad / "adapter_config.json").write_text("{}")
        (ad / "adapter_model.safetensors").write_text("stub")
        z = Path(td) / "sub.zip"
        package_adapter(ad, z, list(req))
        print("package_adapter dry-run OK:", z.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
