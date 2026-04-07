"""Load competition CSVs and build a HuggingFace Dataset for SFT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from datasets import Dataset


REASONING_COLUMNS = ("generated_cot", "cot", "reasoning", "chain_of_thought")


def load_train_dataframe(
    train_csv: Path | str,
    subsample_size: int | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    path = Path(train_csv)
    if not path.is_file():
        raise FileNotFoundError(f"train_csv not found: {path}")

    df = pl.read_csv(path)
    required = {"prompt", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv missing columns {missing}; have {df.columns}")

    if subsample_size is not None and subsample_size < len(df):
        df = df.sample(n=subsample_size, seed=seed)

    return df


def _pick_reasoning(row: dict[str, Any]) -> str:
    for col in REASONING_COLUMNS:
        if col in row and row[col] is not None:
            text = str(row[col]).strip()
            if text:
                return text
    return ""


def rows_to_hf_dataset(df: pl.DataFrame) -> Dataset:
    return Dataset.from_pandas(df.to_pandas())


def attach_training_text(
    hf_ds: Dataset,
    tokenizer,
    user_suffix: str = '\nPut your final answer inside \\boxed{}.',
) -> Dataset:
    """Map rows to a single string field `text` using the tokenizer chat template."""

    def build_training_text(example: dict[str, Any]) -> dict[str, str]:
        prompt = str(example["prompt"])
        answer = str(example["answer"])
        cot = _pick_reasoning(example)

        user_msg = prompt + user_suffix
        if cot:
            assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"
        else:
            assistant_msg = f"\\boxed{{{answer}}}"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            )
        return {"text": text}

    cols = list(hf_ds.column_names)
    return hf_ds.map(build_training_text, remove_columns=cols)
