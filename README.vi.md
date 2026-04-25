# Bộ Skill ML/DL cho AI Agent

> **Ngôn ngữ**: [English](README.md) | **Tiếng Việt**

Bộ sưu tập **15 skill chuyên sâu về Machine Learning & Deep Learning** dành cho AI agent (Claude, Cursor, Cline, hoặc bất kỳ agent nào hỗ trợ chuẩn Agent Skills). Mỗi skill là một module độc lập với hướng dẫn, script template, và tài liệu tham khảo chi tiết.

> **Nguồn gốc**: Bộ skill được trích xuất từ [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) — repo gốc có 125+ skill khoa học, bản này chỉ giữ phần ML/DL relevant. Xem `LICENSE.md` để biết chi tiết license của mỗi skill (chủ yếu Apache-2.0 hoặc MIT).

## Mục lục

- [Cấu trúc của một Skill](#cấu-trúc-của-một-skill)
- [Danh sách 15 skill](#danh-sách-15-skill)
- [Cách dùng](#cách-dùng)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Attribution](#attribution)

## Cấu trúc của một Skill

Mỗi folder skill có cùng anatomy theo chuẩn Anthropic Agent Skills:

```
<skill-name>/
├── SKILL.md            # File chính: YAML frontmatter + Markdown body
├── scripts/            # (Tùy chọn) Python templates chạy được
└── references/         # Tài liệu chi tiết, load on-demand
```

**Progressive Disclosure** — 3 tầng load:

1. **Metadata** (`name` + `description`, ~100 từ): luôn nằm trong context để AI quyết định có trigger skill không.
2. **Body `SKILL.md`** (<500 dòng): chỉ load khi skill được trigger.
3. **`references/` + `scripts/`**: load on-demand khi AI cần đào sâu.

Ví dụ frontmatter chuẩn (`pytorch-lightning/SKILL.md`):

```yaml
---
name: pytorch-lightning
description: Deep learning framework (PyTorch Lightning). Organize PyTorch code into LightningModules, configure Trainers for multi-GPU/TPU, implement data pipelines, callbacks, logging (W&B, TensorBoard), distributed training (DDP, FSDP, DeepSpeed), for scalable neural network training.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---
```

## Danh sách 15 skill

### Deep Learning

| Skill | Mô tả | Use case chính |
|---|---|---|
| `pytorch-lightning` | Framework chuẩn hoá PyTorch code | Train neural network, multi-GPU, DDP/FSDP/DeepSpeed |
| `transformers` | HuggingFace Transformers | Pre-trained NLP/CV/audio models, fine-tune |
| `torch-geometric` | Graph Neural Networks | GCN, GAT, GraphSAGE trên dữ liệu đồ thị |

### Classical ML

| Skill | Mô tả | Use case chính |
|---|---|---|
| `scikit-learn` | Thư viện ML cổ điển toàn diện | Classification, regression, clustering, pipeline |
| `scikit-survival` | Survival analysis | Cox model, Kaplan-Meier, time-to-event |
| `shap` | Explainable AI (SHapley values) | Feature importance, model debugging, fairness |

### Time Series & Statistics

| Skill | Mô tả | Use case chính |
|---|---|---|
| `aeon` | Time series toolkit | Forecasting, classification, clustering chuỗi thời gian |
| `statsmodels` | Statistical modeling | OLS, GLM, ARIMA, hypothesis testing |
| `pymc` | Bayesian modeling | MCMC, probabilistic programming |

### Reinforcement Learning

| Skill | Mô tả | Use case chính |
|---|---|---|
| `stable-baselines3` | RL algorithms (PPO, SAC, DQN, A2C) | Train agent trên Gymnasium environment |
| `pufferlib` | High-performance RL framework | Multi-agent, vectorized envs, throughput cao |

### Tools & Infrastructure

| Skill | Mô tả | Use case chính |
|---|---|---|
| `umap-learn` | Dimensionality reduction | Visualization, embedding analysis |
| `pymoo` | Multi-objective optimization | NSGA-II, MOEA, Pareto frontier |
| `optimize-for-gpu` | GPU acceleration toolkit | CuPy, Numba CUDA, RAPIDS (cuDF, cuML, cuGraph) |
| `modal` | Serverless cloud compute | GPU training trên cloud, scale on-demand |

## Cách dùng

### Option 1 — Cài làm skill cho Claude Code

Copy folder skill vào thư mục skill của Claude:

```pwsh
# Windows (pwsh) — chạy từ thư mục gốc của repo
Copy-Item -Recurse ".\scientific-skills\pytorch-lightning" "$HOME\.claude\skills\"
```

```bash
# Linux / macOS — chạy từ thư mục gốc của repo
cp -r ./scientific-skills/pytorch-lightning ~/.claude/skills/
```

Sau đó Claude sẽ tự động discover skill khi câu hỏi của bạn liên quan đến PyTorch Lightning (vd: *"giúp tôi setup multi-GPU training"*).

### Option 2 — Đọc trực tiếp làm tài liệu tham khảo

Mở `SKILL.md` của skill cần học. Ví dụ:

- `scientific-skills/pytorch-lightning/SKILL.md` — overview + workflow nhanh.
- `scientific-skills/pytorch-lightning/references/distributed_training.md` — đào sâu DDP/FSDP/DeepSpeed.
- `scientific-skills/pytorch-lightning/scripts/template_lightning_module.py` — boilerplate Python copy-paste được.

### Option 3 — Dùng làm template để viết skill mới của bạn

Copy cấu trúc folder của bất kỳ skill nào, sửa lại `SKILL.md` theo workflow của bạn:

```
my-custom-skill/
├── SKILL.md            # frontmatter + body theo workflow của bạn
├── scripts/
│   └── my_template.py
└── references/
    ├── concept_a.md
    └── concept_b.md
```

3 best practice cốt lõi khi viết skill mới (theo Anthropic):

- **Description phải hơi "pushy"** — kèm các từ khoá user hay dùng (`fine-tune`, `GPU training`, `explain prediction`...) để skill không bị under-trigger.
- **Body <500 dòng** — chi tiết kỹ thuật đẩy xuống `references/`.
- **Imperative form** — "Load the model with..." thay vì "You can load...".

## Tài liệu tham khảo

Trong `docs/` (đã clone về sẵn):

- **`scientific-skills.md`** (~104 KB) — Mô tả chi tiết toàn bộ 125+ skill từ repo gốc (cả các skill bioinformatics, cheminformatics, clinical... chưa clone về).
- **`examples.md`** (~114 KB) — Workflow ví dụ end-to-end kết hợp nhiều skill (drug discovery pipeline, single-cell RNA-seq, multi-omics biomarker discovery...).

Liên kết ngoài:

- [Anthropic Skills Documentation](https://docs.claude.com/en/api/skills-guide) — chuẩn chính thức.
- [skill-creator (Anthropic)](https://github.com/anthropics/skills/tree/main/skills/skill-creator) — skill chính thức của Anthropic để tạo skill mới + chạy eval.
- [The Complete Guide to Building Skills for Claude (PDF)](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) — guide chính thức.
- [K-Dense-AI repo gốc](https://github.com/K-Dense-AI/scientific-agent-skills) — luôn cập nhật version mới.

## Attribution

Toàn bộ 15 skill trong `scientific-skills/` được phát triển và duy trì bởi **K-Dense Inc.** Mỗi skill có license riêng (chủ yếu Apache-2.0 hoặc MIT), xem `LICENSE.md` ở root và phần `metadata` trong frontmatter của từng `SKILL.md` để biết chi tiết.

Bản README này được viết riêng cho thư mục local, không thuộc repo gốc.
