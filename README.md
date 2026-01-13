# Repo-JEPA: Semantic Code Navigator

A **Joint Embedding Predictive Architecture** for semantic code search on consumer hardware.

## 🎯 What It Does

Query with natural language ("handle login failure") → Get the exact function, even if keywords are missing.

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Code Encoder   │     │ Docstring Enc.  │
│  (Trainable)    │     │  (EMA Target)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
     Code Embed.            Doc Embed.
         │                       │
         └───────────┬───────────┘
                     ▼
              VICReg Loss
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run sanity check
python -m src.train --sanity-check

# Full training
python -m src.train --dataset codesearchnet --epochs 10

# Evaluate
python -m src.eval.code_search --checkpoint checkpoints/best.pt
```

## 📊 Benchmarks

| Metric | Target | Description |
|--------|--------|-------------|
| MRR | > 0.6 | Mean Reciprocal Rank on CodeSearchNet |
| Linear Probe | > 85% | Code intent classification accuracy |

## 🔧 Training Hardware

- **Minimum**: RTX 3060 (12GB) / M1 Mac (16GB)
- **Recommended**: RTX 3090 (24GB) / RunPod
- **VRAM Usage**: ~4-6GB for 110M params

## 📁 Project Structure

```
repo-jepa/
├── src/
│   ├── config.py      # Model configuration
│   ├── model.py       # RepoJEPA architecture
│   ├── loss.py        # VICReg loss
│   ├── train.py       # Training loop
│   ├── data/          # Data loaders
│   └── eval/          # Validation scripts
├── hf_export/         # Hugging Face export
└── tests/             # Unit tests
```

## 📜 License

MIT
