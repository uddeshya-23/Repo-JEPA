---
language: en
tags:
- code
- semantic-search
- jepa
- code-search
license: mit
datasets:
- claudios/code_search_net
metrics:
- mrr
---

# Repo-JEPA: Semantic Code Navigator (SOTA 0.90 MRR)

A **Joint Embedding Predictive Architecture** (JEPA) for semantic code search, trained on 411,000 real Python functions using an NVIDIA H100.

## 🏆 Performance

Tested on 1,000 unseen real-world Python functions from CodeSearchNet.

| Metric | Result | Target |
|--------|--------|--------|
| **MRR** | **0.9052** | 0.60 |
| **Hits@1** | **86.2%** | - |
| **Hits@5** | **95.9%** | - |
| **Hits@10** | **97.3%** | - |
| **Median Rank** | **1.0** | - |

## 🧩 Usage (AutoModel)

```python
from transformers import AutoModel, AutoTokenizer

# 1. Load Model
model = AutoModel.from_pretrained("uddeshya-k/RepoJepa", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")


# 2. Encode Code
code = "def handle_login(user): return auth.verify(user)"
code_embed = model.encode_code(**tokenizer(code, return_tensors="pt"))

# 3. Encode Query
query = "how to authenticate users?"
query_embed = model.encode_query(**tokenizer(query, return_tensors="pt"))

# 4. Search
similarity = (code_embed @ query_embed.T).item()
print(f"Similarity: {similarity:.4f}")
```

## 🏗️ Technical Details

- **Backbone**: CodeBERT (RoBERTa-style)
- **Loss**: VICReg (Variance-Invariance-Covariance Regularization)
- **Hardware**: NVIDIA H100 PCIe (80GB VRAM)
- **Optimizer**: AdamW + OneCycleLR
