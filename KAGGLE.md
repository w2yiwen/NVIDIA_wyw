# 在 Kaggle GPU 上运行（关联本仓库）

## 1. 准备

1. **加入竞赛** [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)（或当前竞赛页），否则无法挂载官方 `train.csv`。
2. **新建 Notebook**：右上角 **Create → New Notebook**。
3. **加速器**：Session options 里选 **GPU**（建议 **H100** 或可用最强 GPU）。
4. **Internet**：Settings → Internet **ON**（用于 `git clone` 与 `kagglehub` 拉模型）。
5. **Kaggle API**（拉 Hub 模型常用）：Settings → Add Secrets，添加与本地 `kaggle.json` 一致的 **KAGGLE_USERNAME** / **KAGGLE_KEY**（若 `kagglehub` 报错再配）。

## 2. 推荐：Notebook 里直接 clone 本仓库

上传本仓库中的 **`notebooks/kaggle_train.ipynb`** 到 Kaggle，按顺序运行全部单元；或在空 Notebook 里复制该 notebook 的代码。

逻辑简述：

- `git clone https://github.com/w2yiwen/NVIDIA_wyw.git` → `/kaggle/working/NVIDIA_wyw`
- 设置 `NEMOTRON_WORKSPACE` 与 `PYTHONPATH`
- `pip install` 缺的依赖（`kagglehub`、`trl`、`peft` 等）
- `python -m src.train --config configs/kaggle.yaml --kaggle_bootstrap`
- `python -m src.package_submission --config configs/kaggle.yaml`
- 在 Output 里下载 **`submission.zip`**

`configs/kaggle.yaml` 已写好：

- 基座缓存：`/kaggle/working/cache`（HF + Kaggle Hub）
- 训练数据：`/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv`
- 适配器输出：`/kaggle/working/adapter`
- 提交 zip：`/kaggle/working/submission.zip`

## 3. 可选：把代码打成 Kaggle Dataset

若不想每次 clone：

1. 将本仓库打包上传为 **New Dataset**，例如 slug `nvidia-wyw-code`。
2. Notebook → **Add data** 挂载该 dataset。
3. 把 notebook 里 `REPO_DIR` 改为 `/kaggle/input/nvidia-wyw-code`（以你实际路径为准），并 **不要** 再执行 `git clone` 单元。

## 4. 离线包（与官方教程一致）

若竞赛提供 **NVIDIA offline packages** 等 Input，可与 `kaggle_bootstrap` 中的路径对齐；默认会先尝试离线目录，不存在则回退 PyPI。

## 5. 提交

训练结束后在 `/kaggle/working/` 找到 **`submission.zip`**，到竞赛页 **Submit** 上传。

---

English summary: enable **GPU + Internet**, run `notebooks/kaggle_train.ipynb` (or equivalent cells), which clones this repo, trains with `configs/kaggle.yaml` and `--kaggle_bootstrap`, then builds `submission.zip` under `/kaggle/working/`.
