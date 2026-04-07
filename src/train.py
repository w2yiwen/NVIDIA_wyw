#!/usr/bin/env python3
"""Fine-tune Nemotron-3-Nano-30B with LoRA (PEFT + TRL SFT). Competition rank ≤ 32."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(argv: list[str] | None = None) -> None:
    """CLI entry; pass `argv` from a notebook (e.g. `run(['--kaggle_bootstrap', ...])`)."""
    # Lazy imports so `python -m src.package_submission` stays light on import.
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from src.data_pipeline import attach_training_text, load_train_dataframe, rows_to_hf_dataset
    from src.paths import apply_disk_cache_env, resolve_path

    parser = argparse.ArgumentParser(description="LoRA SFT for Nemotron reasoning challenge")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train_csv", type=str, default=None, help="Override train CSV path")
    parser.add_argument("--model_path", type=str, default=None, help="Local model dir or leave default for Hub download")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--weights_disk", type=str, default=None, help="Override paths.weights_disk_root (HF + Kaggle cache)")
    parser.add_argument("--kaggle_bootstrap", action="store_true", help="Run offline pip + Triton hacks (Kaggle only)")
    args = parser.parse_args(argv)

    cfg_path = resolve_path(args.config)
    cfg = _load_config(cfg_path)

    disk_root = args.weights_disk or (cfg.get("paths") or {}).get("weights_disk_root")
    if disk_root:
        apply_disk_cache_env(disk_root)

    if args.kaggle_bootstrap:
        from src.kaggle_bootstrap import apply_triton_ptxas_hack, install_offline_packages

        install_offline_packages()
        apply_triton_ptxas_hack()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    train_csv = resolve_path(args.train_csv or cfg["data"]["train_csv"])
    subsample = cfg["data"].get("subsample_size")
    df = load_train_dataframe(train_csv, subsample_size=subsample)
    hf_raw = rows_to_hf_dataset(df)

    model_path = args.model_path
    if not model_path:
        local_cfg = (cfg.get("model") or {}).get("local_path")
        if local_cfg:
            model_path = str(resolve_path(local_cfg))
    if not model_path:
        mid = cfg["model"]["kaggle_model_id"]
        try:
            import kagglehub

            model_path = kagglehub.model_download(mid)
        except Exception as e:
            raise RuntimeError(
                f"Set --model_path or model.local_path in config, or install kagglehub and use kaggle_model_id ({mid}). Original error: {e}"
            ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_dataset = attach_training_text(hf_raw, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    if args.kaggle_bootstrap:
        from src.kaggle_bootstrap import patch_nemotron_fast_path_disabled, patch_triton_compiler_version

        patch_nemotron_fast_path_disabled()
        patch_triton_compiler_version()

    t = cfg["training"]
    rank = int(t["lora_rank"])
    if rank > 32:
        raise ValueError("Competition allows max LoRA rank 32.")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=int(t["lora_alpha"]),
        target_modules="all-linear",
        lora_dropout=float(t["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    output_dir = resolve_path(args.output_dir or t["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(t["batch_size"]),
        gradient_accumulation_steps=int(t["grad_accum"]),
        num_train_epochs=float(t["num_epochs"]),
        learning_rate=float(t["learning_rate"]),
        logging_steps=int(t["logging_steps"]),
        bf16=True,
        max_grad_norm=float(t["max_grad_norm"]),
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=float(t["warmup_ratio"]),
        save_strategy=str(t.get("save_strategy", "no")),
        report_to="none",
        dataset_text_field="text",
        max_length=int(t["max_seq_len"]),
        packing=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Adapter and tokenizer saved to {output_dir}")


def main() -> None:
    run(None)


if __name__ == "__main__":
    main()
