## 🧩 Usage for Others (Inference)

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
