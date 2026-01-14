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

## 🧩 Usage for Others (Inference)

If you just want to use the model for semantic search in your own project:

```python
from src.utils.search import RepoJEPASearch

# 1. Initialize (will download from Hugging Face)
searcher = RepoJEPASearch("uddeshya-23/repo-jepa")

# 2. Index your code repository
searcher.add_code([
    "def calculate_tax(amount): return amount * 0.2",
    "def auth_user(token): return db.find(token)",
    "def save_log(msg): print(f'[LOG] {msg}')"
])

# 3. Query with natural language
results = searcher.query("how to pay taxes?", top_k=1)
print(results[0][0])  # Prints the first code snippet
```

## 📊 Performance (H100 Result)

| Metric | Result | Target |
|--------|--------|--------|
| **MRR** | **0.9052** | 0.60 |
| Hits@1 | 86.2% | - |
| Median Rank | 1.0 | - |

## 🔧 Training Hardware

- **Minimum**: RTX 3060 (12GB) / M1 Mac (16GB)
- **Recommended**: RTX 3090 (24GB) / RunPod
- **VRAM Usage**: ~4-6GB for 110M params

## 📁 Project Structure

```
repo-jepa/
├── src/
│   ├── model.py       # Dual-encoder architecture
│   ├── train.py       # Training with checkpoint resume
│   ├── data/          # Real-data loaders (CodeSearchNet)
│   ├── eval/          # MRR benchmarks
│   └── utils/         
│       └── search.py  # User-friendly Search Engine
├── hf_export/         # Tools to export to Hugging Face
└── notebooks/         # Analysis and demos
```

## 📜 License

MIT
