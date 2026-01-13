"""
Upload Repo-JEPA to Hugging Face Hub.

Usage:
    python upload.py --repo your-username/repo-jepa-110m --checkpoint ../checkpoints/best.pt
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo


def convert_checkpoint_to_hf(checkpoint_path: str, output_dir: str):
    """
    Convert training checkpoint to Hugging Face format.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract model state dict
    state_dict = checkpoint["model_state_dict"]
    
    # Save as safetensors (recommended) or pytorch_model.bin
    try:
        from safetensors.torch import save_file
        save_file(state_dict, output_path / "model.safetensors")
        print("Saved model.safetensors")
    except ImportError:
        torch.save(state_dict, output_path / "pytorch_model.bin")
        print("Saved pytorch_model.bin (install safetensors for better format)")
    
    # Copy config and modeling files
    script_dir = Path(__file__).parent
    shutil.copy(script_dir / "config.json", output_path / "config.json")
    shutil.copy(script_dir / "modeling_repo_jepa.py", output_path / "modeling_repo_jepa.py")
    
    # Create model card
    model_card = """---
language: en
tags:
- code
- semantic-search
- jepa
- code-search
license: mit
datasets:
- code_search_net
---

# Repo-JEPA: Semantic Code Navigator

A **Joint Embedding Predictive Architecture** for semantic code search.

## 🎯 What It Does

Query with natural language ("handle login failure") → Get the exact function, even if keywords are missing.

## Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("your-username/repo-jepa-110m", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Encode code
code = "def add(a, b): return a + b"
inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
code_embedding = model(**inputs)

# Encode query
query = "function that adds two numbers"
query_inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
query_embedding = model(**query_inputs)

# Compute similarity
similarity = (code_embedding @ query_embedding.T).item()
```

## Architecture

- **Base**: CodeBERT (RoBERTa-style, 110M params)
- **Training**: VICReg loss (Variance-Invariance-Covariance)
- **Dataset**: CodeSearchNet (Python)

## Benchmarks

| Metric | Score |
|--------|-------|
| MRR | TBD |
| Hits@1 | TBD |
| Hits@10 | TBD |

## Training

Trained using the JEPA framework with EMA target encoder updates.

## License

MIT
"""
    
    with open(output_path / "README.md", "w") as f:
        f.write(model_card)
    
    print(f"Export complete! Files saved to {output_path}")
    return output_path


def upload_to_hub(folder_path: str, repo_id: str, private: bool = False):
    """Upload model to Hugging Face Hub."""
    api = HfApi()
    
    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, private=private, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload
    print(f"Uploading to {repo_id}...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"✅ Upload complete! Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload Repo-JEPA to Hugging Face")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo ID (username/model-name)")
    parser.add_argument("--output-dir", type=str, default="./hf_model", help="Temporary output directory")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--skip-upload", action="store_true", help="Only convert, don't upload")
    
    args = parser.parse_args()
    
    # Convert checkpoint
    output_path = convert_checkpoint_to_hf(args.checkpoint, args.output_dir)
    
    # Upload if requested
    if not args.skip_upload:
        upload_to_hub(str(output_path), args.repo, args.private)


if __name__ == "__main__":
    main()
