#!/usr/bin/env bash
# 全流程：训练（权重缓存与输出在 /root/autodl-tmp/.autodl）→ 打 submission.zip
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export NEMOTRON_WORKSPACE="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p /root/autodl-tmp/.autodl

CONFIG="${1:-configs/autodl.yaml}"
cd "$ROOT"

echo "=== Train (config: $CONFIG) ==="
python -m src.train --config "$CONFIG"

echo "=== Package submission.zip ==="
python -m src.package_submission --config "$CONFIG"

echo "Done. Adapter + zip are under /root/autodl-tmp/.autodl/runs/nemotron_lora/"
