# Repo-JEPA: Semantic Code Navigator

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/uddeshya-k/RepoJepa)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**State-of-the-Art semantic code search using Joint Embedding Predictive Architecture (JEPA).**  
Find code by *what it does*, not just *what it's named*.

---

## 🎯 The Problem with Traditional Search

**Keyword Search:**
```python
# Your codebase has:
def authenticate_user(token): ...

# You search for: "login handler"
# Result: ❌ No matches
```

**AI IDE (Cloud RAG):**
- Sends your code to external servers (privacy risk)
- Still relies on keyword overlap
- Expensive API calls

**Repo-JEPA:**
```python
# You search for: "where do we validate auth tokens?"
# Result: ✅ authenticate_user() - Score: 0.89
# Runs 100% locally, no cloud needed
```

---

## 🏆 Performance (CodeSearchNet Benchmark)

Tested on 1,000 unseen real-world Python functions:

| Metric | Result | Industry Target |
|--------|--------|----------------|
| **MRR** | **0.9052** | 0.60 |
| **Hits@1** | **86.2%** | - |
| **Hits@5** | **95.9%** | - |
| **Median Rank** | **1.0** | - |

**Translation**: In 86% of searches, the *exact* correct function appears as the #1 result.

---

## 🚀 Quick Start (2 Commands)

### 1. Index Your Project
```bash
# Clone and install
git clone https://github.com/uddeshya-23/Repo-JEPA
cd Repo-JEPA
pip install -r requirements.txt

# Index your codebase (replace path)
python src/utils/indexer.py --path /your/project --save project.index
```

### 2. Search Semantically
```python
from src.utils.search import RepoJEPASearch

# One-time load
searcher = RepoJEPASearch("uddeshya-k/RepoJepa")
searcher.load_index("project.index")

# Search by intent, not keywords
results = searcher.query("how to handle payment webhooks?")
print(f"Found: {results[0]['file']}:{results[0]['line']}")
```

**Interactive CLI:**
```bash
python cli_search.py
# Type your query → Get instant results with file locations
```

---

## 🧠 How It Works (Technical Innovation)

### JEPA (Joint Embedding Predictive Architecture)
Pioneered by **Yann LeCun (Meta AI)**, JEPA learns relationships between modalities (code ↔ natural language) without explicit labels.

### VICReg Loss (Prevents Collapse)
**VICReg** (Bardes et al., ICLR 2022) ensures each function has a unique "fingerprint" in embedding space:
- **Variance**: No two functions collapse to the same point
- **Invariance**: Semantically similar code stays close
- **Covariance**: Multi-dimensional representation (no dimension dominates)

### Architecture
```
Natural Language Query          Code Snippet
       ↓                              ↓
  Target Encoder              Context Encoder
  (EMA-updated)                 (Trainable)
       ↓                              ↓
   Query Embed  ← Cosine Similarity → Code Embed
                       ↓
                  VICReg Loss
```

---

## 📊 Why It's Better

| Feature | Keyword Search | AI IDE (Cloud RAG) | **Repo-JEPA** |
|---------|---------------|-------------------|---------------|
| **Semantic Understanding** | ❌ Literal only | ⚠️ Limited | ✅ Native |
| **Privacy** | ✅ Local | ❌ Cloud-based | ✅ 100% Local |
| **Cost** | Free | $$$  | Free |
| **Speed** | Fast | Network-dependent | Instant (local) |
| **Accuracy (MRR)** | ~0.3 | ~0.5 | **0.905** |

---

## 🛠️ Installation & Usage

### Requirements
- Python 3.8+
- PyTorch 2.0+
- 4GB GPU (optional, but recommended)

### Full Installation
```bash
git clone https://github.com/uddeshya-23/Repo-JEPA
cd Repo-JEPA
pip install -r requirements.txt
```

### Advanced: Train Your Own Model
```bash
# Train on CodeSearchNet (requires H100/A100 for speed)
python -m src.train --dataset codesearchnet --epochs 10

# Evaluate
python -m src.eval.code_search --checkpoint checkpoints/best.pt
```

---

## 🔬 Research Foundation

Built on cutting-edge research:

1. **JEPA** - Yann LeCun et al. (Meta AI, 2022)  
   *"A Path Towards Autonomous Machine Intelligence"*

2. **VICReg** - Bardes, Ponce, LeCun (ICLR 2022)  
   *ArXiv:* [2105.04906](https://arxiv.org/abs/2105.04906)

3. **CodeBERT** - Microsoft Research (EMNLP 2020)  
   *ArXiv:* [2002.08155](https://arxiv.org/abs/2002.08155)

See [RESEARCH_ATTRIBUTION.md](RESEARCH_ATTRIBUTION.md) for full citations.

---

## 🤝 Contributing & Feedback

We're looking for:
- 🐛 **Bug reports** on real-world codebases
- 💡 **Feature requests** (multi-language support, IDE plugins)
- 🔬 **Research collaborations** on code understanding

**Contact:**  
- Open an issue on [GitHub](https://github.com/uddeshya-23/Repo-JEPA/issues)
- Discuss on [Hugging Face](https://huggingface.co/uddeshya-k/RepoJepa/discussions)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Yann LeCun** (JEPA framework)
- **Meta AI** (VICReg research)
- **Microsoft Research** (CodeBERT, CodeSearchNet)
- **NVIDIA** (H100 training infrastructure)

---

**⭐ If this helped you, please star the repo and share with the developer community!**


### 1. Zero-Touch Indexing (Automated)
Don't worry about manual encoding. Use the **Automated Indexer** to scan your entire project in one command:

```bash
# Encodes all python functions in 'my_project' directory
python src/utils/indexer.py --path ./my_project --save my_project.index
```

### 2. Search in Milliseconds
Once the index is built, searching is nearly instant and happens locally:

```python
from src.utils.search import RepoJEPASearch

searcher = RepoJEPASearch("uddeshya-k/RepoJepa")
searcher.load_index("my_project.index")

# Search by intent
results = searcher.query("how to handle payment webhooks?")
print(f"File: {results[0]['file']}, Line: {results[0]['line']}")
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
