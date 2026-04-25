# ML/DL Skills for AI Agents

> **Language**: **English** | [Tiếng Việt](README.vi.md)

A curated collection of **15 specialized Machine Learning & Deep Learning skills** for AI agents (Claude, Cursor, Cline, or any agent that supports the Agent Skills standard). Each skill is a self-contained module with instructions, executable script templates, and detailed reference documentation.

> **Source**: Extracted from [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) — the upstream repo contains 125+ scientific skills; this collection retains only the ML/DL-relevant ones. See `LICENSE.md` for per-skill license details (mostly Apache-2.0 or MIT).

## Table of Contents

- [Skill Anatomy](#skill-anatomy)
- [The 15 Skills](#the-15-skills)
- [How to Use](#how-to-use)
- [References](#references)
- [Attribution](#attribution)

## Skill Anatomy

Each skill folder follows the Anthropic Agent Skills convention:

```
<skill-name>/
├── SKILL.md            # Main file: YAML frontmatter + Markdown body
├── scripts/            # (Optional) Runnable Python templates
└── references/         # Detailed docs, loaded on-demand
```

**Progressive Disclosure** — three loading tiers:

1. **Metadata** (`name` + `description`, ~100 words): always in context so the AI can decide whether to trigger the skill.
2. **`SKILL.md` body** (<500 lines ideal): loaded only when the skill triggers.
3. **`references/` + `scripts/`**: loaded on-demand when the AI needs deeper detail.

Standard frontmatter example (`pytorch-lightning/SKILL.md`):

```yaml
---
name: pytorch-lightning
description: Deep learning framework (PyTorch Lightning). Organize PyTorch code into LightningModules, configure Trainers for multi-GPU/TPU, implement data pipelines, callbacks, logging (W&B, TensorBoard), distributed training (DDP, FSDP, DeepSpeed), for scalable neural network training.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---
```

## The 15 Skills

### Deep Learning

| Skill | Description | Primary Use Case |
|---|---|---|
| `pytorch-lightning` | PyTorch boilerplate-elimination framework | Train neural networks, multi-GPU, DDP/FSDP/DeepSpeed |
| `transformers` | HuggingFace Transformers | Pre-trained NLP/CV/audio models, fine-tuning |
| `torch-geometric` | Graph Neural Networks | GCN, GAT, GraphSAGE on graph data |

### Classical ML

| Skill | Description | Primary Use Case |
|---|---|---|
| `scikit-learn` | Comprehensive classical ML library | Classification, regression, clustering, pipelines |
| `scikit-survival` | Survival analysis | Cox model, Kaplan-Meier, time-to-event |
| `shap` | Explainable AI (SHapley values) | Feature importance, model debugging, fairness |

### Time Series & Statistics

| Skill | Description | Primary Use Case |
|---|---|---|
| `aeon` | Time series toolkit | Forecasting, classification, clustering |
| `statsmodels` | Statistical modeling | OLS, GLM, ARIMA, hypothesis testing |
| `pymc` | Bayesian modeling | MCMC, probabilistic programming |

### Reinforcement Learning

| Skill | Description | Primary Use Case |
|---|---|---|
| `stable-baselines3` | RL algorithms (PPO, SAC, DQN, A2C) | Train agents on Gymnasium environments |
| `pufferlib` | High-performance RL framework | Multi-agent, vectorized envs, high throughput |

### Tools & Infrastructure

| Skill | Description | Primary Use Case |
|---|---|---|
| `umap-learn` | Dimensionality reduction | Visualization, embedding analysis |
| `pymoo` | Multi-objective optimization | NSGA-II, MOEA, Pareto frontier |
| `optimize-for-gpu` | GPU acceleration toolkit | CuPy, Numba CUDA, RAPIDS (cuDF, cuML, cuGraph) |
| `modal` | Serverless cloud compute | GPU training in the cloud, scale on-demand |

## How to Use

### Option 1 — Install as a Claude Code skill

Copy the skill folder into Claude's skills directory:

```pwsh
# Windows (pwsh) — run from the repo root
Copy-Item -Recurse ".\scientific-skills\pytorch-lightning" "$HOME\.claude\skills\"
```

```bash
# Linux / macOS — run from the repo root
cp -r ./scientific-skills/pytorch-lightning ~/.claude/skills/
```

Claude will then auto-discover the skill when your prompt is relevant (e.g., *"help me set up multi-GPU training"*).

### Option 2 — Read directly as documentation

Open the `SKILL.md` of the skill you want to learn. For example:

- `scientific-skills/pytorch-lightning/SKILL.md` — overview + quick workflow.
- `scientific-skills/pytorch-lightning/references/distributed_training.md` — deep dive on DDP / FSDP / DeepSpeed.
- `scientific-skills/pytorch-lightning/scripts/template_lightning_module.py` — copy-paste-ready Python boilerplate.

### Option 3 — Use as a template for your own skills

Copy the folder structure of any skill and edit `SKILL.md` to match your workflow:

```
my-custom-skill/
├── SKILL.md            # frontmatter + body for your workflow
├── scripts/
│   └── my_template.py
└── references/
    ├── concept_a.md
    └── concept_b.md
```

Three core best practices for writing new skills (per Anthropic):

- **Make the description slightly "pushy"** — include keywords users actually say (`fine-tune`, `GPU training`, `explain prediction`...) so the skill doesn't under-trigger.
- **Keep the body under 500 lines** — push detailed technical content into `references/`.
- **Use the imperative form** — "Load the model with..." rather than "You can load...".

## References

In `docs/` (already cloned):

- **`scientific-skills.md`** (~104 KB) — Detailed description of all 125+ skills from the source repo (including bioinformatics, cheminformatics, clinical skills not cloned here).
- **`examples.md`** (~114 KB) — End-to-end workflow examples combining multiple skills (drug discovery pipeline, single-cell RNA-seq, multi-omics biomarker discovery, etc.).

External links:

- [Anthropic Skills Documentation](https://docs.claude.com/en/api/skills-guide) — official spec.
- [skill-creator (Anthropic)](https://github.com/anthropics/skills/tree/main/skills/skill-creator) — Anthropic's official skill for authoring and evaluating new skills.
- [The Complete Guide to Building Skills for Claude (PDF)](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf) — official guide.
- [K-Dense-AI source repo](https://github.com/K-Dense-AI/scientific-agent-skills) — for upstream updates.

## Attribution

All 15 skills in `scientific-skills/` were developed and are maintained by **K-Dense Inc.** Each skill has its own license (mostly Apache-2.0 or MIT); see `LICENSE.md` at the root and the `metadata` section of each `SKILL.md`'s frontmatter for details.

This README was written specifically for this local directory and is not part of the upstream repo.
