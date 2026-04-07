# NVIDIA_wyw

NVIDIA **Nemotron Model Reasoning Challenge** — LoRA 微调工程：从 `train.csv` 训练 **Nemotron-3-Nano-30B** 适配器，并打包 **`submission.zip`**（`adapter_config.json` + `adapter_model.safetensors` 等）。

仓库地址：**https://github.com/w2yiwen/NVIDIA_wyw**

## Kaggle GPU（推荐）

1. 阅读 **[KAGGLE.md](./KAGGLE.md)**（一步步：开 GPU、开网、挂载竞赛数据）。
2. 上传并运行 **`notebooks/kaggle_train.ipynb`**：自动 `git clone` 本仓库 → 训练 → 生成 `/kaggle/working/submission.zip`。

核心配置：**[configs/kaggle.yaml](./configs/kaggle.yaml)**（数据路径、缓存与输出均在 `/kaggle/working`）。

## 本地 / AutoDL

- 默认配置：`configs/default.yaml`
- 数据盘 + 大缓存：**`configs/autodl.yaml`** + `./scripts/run_full_autodl.sh`

详见原 README 小节与 `requirements.txt`。

## 目录结构

| 路径 | 说明 |
|------|------|
| `src/train.py` | 训练入口；`run([...])` 供 Notebook 调用 |
| `src/package_submission.py` | 打 `submission.zip` |
| `src/data_pipeline.py` | CSV → HF Dataset，`\boxed{}` 格式 |
| `src/kaggle_bootstrap.py` | Kaggle 离线包 + Triton/ptxas 补丁 |
| `configs/kaggle.yaml` | Kaggle 路径 |
| `configs/autodl.yaml` | 本地数据盘路径示例 |

## 规则摘要

- LoRA **rank ≤ 32**（默认 32）。
- 提交物为 **`submission.zip`**，内含竞赛要求的 LoRA 文件。

## Clone

```bash
git clone https://github.com/w2yiwen/NVIDIA_wyw.git
cd NVIDIA_wyw
pip install -r requirements.txt
```
