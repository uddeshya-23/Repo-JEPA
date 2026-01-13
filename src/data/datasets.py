"""
CodeSearchNet Dataset Loader

Loads code-docstring pairs for training Repo-JEPA.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, DatasetDict


@dataclass
class CodeDocPair:
    """A single code-documentation pair."""
    code: str
    docstring: str
    func_name: str
    language: str = "python"


class CodeSearchNetDataset(Dataset):
    """
    CodeSearchNet dataset for semantic code search.
    
    Each sample contains:
        - code: The function body
        - docstring: The documentation string
        - func_name: Function name (can be used for additional supervision)
    """
    
    def __init__(
        self,
        split: str = "train",
        language: str = "python",
        max_samples: Optional[int] = None,
        min_doc_length: int = 10,
        min_code_length: int = 20,
    ):
        self.split = split
        self.language = language
        self.min_doc_length = min_doc_length
        self.min_code_length = min_code_length
        
        # Load dataset
        self.data = self._load_and_filter(max_samples)
    
    def _load_and_filter(self, max_samples: Optional[int]) -> List[CodeDocPair]:
        """Load and filter CodeSearchNet data."""
        print(f"Loading CodeSearchNet ({self.language}/{self.split})...")
        
        try:
            dataset = load_dataset(
                "code_search_net",
                self.language,
                split=self.split,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to mock data for testing...")
            return self._create_mock_data(max_samples or 100)
        
        pairs = []
        for item in dataset:
            docstring = item.get("func_documentation_string", "")
            code = item.get("func_code_string", "")
            func_name = item.get("func_name", "unknown")
            
            # Filter by length
            if len(docstring) < self.min_doc_length:
                continue
            if len(code) < self.min_code_length:
                continue
            
            pairs.append(CodeDocPair(
                code=code,
                docstring=docstring,
                func_name=func_name,
                language=self.language,
            ))
            
            if max_samples and len(pairs) >= max_samples:
                break
        
        print(f"Loaded {len(pairs)} code-doc pairs")
        return pairs
    
    def _create_mock_data(self, n_samples: int) -> List[CodeDocPair]:
        """Create mock data for testing without internet."""
        mock_functions = [
            ("def add(a, b):\n    return a + b", "Add two numbers together."),
            ("def multiply(x, y):\n    return x * y", "Multiply two values."),
            ("def read_file(path):\n    with open(path) as f:\n        return f.read()", "Read contents of a file."),
            ("def write_json(data, path):\n    import json\n    with open(path, 'w') as f:\n        json.dump(data, f)", "Write data to JSON file."),
            ("def fetch_url(url):\n    import requests\n    return requests.get(url).text", "Fetch content from URL."),
            ("def sort_list(items):\n    return sorted(items)", "Sort a list in ascending order."),
            ("def filter_none(items):\n    return [x for x in items if x is not None]", "Remove None values from list."),
            ("def calculate_mean(numbers):\n    return sum(numbers) / len(numbers)", "Calculate arithmetic mean."),
            ("def validate_email(email):\n    import re\n    return bool(re.match(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', email))", "Validate email format."),
            ("def hash_password(password):\n    import hashlib\n    return hashlib.sha256(password.encode()).hexdigest()", "Hash password using SHA256."),
        ]
        
        pairs = []
        for i in range(n_samples):
            code, doc = mock_functions[i % len(mock_functions)]
            pairs.append(CodeDocPair(
                code=code,
                docstring=doc,
                func_name=f"func_{i}",
                language="python",
            ))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        pair = self.data[idx]
        return {
            "code": pair.code,
            "docstring": pair.docstring,
            "func_name": pair.func_name,
        }


def load_codesearchnet(
    language: str = "python",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Dict[str, CodeSearchNetDataset]:
    """
    Load CodeSearchNet train and validation splits.
    
    Returns:
        Dict with 'train' and 'validation' datasets
    """
    return {
        "train": CodeSearchNetDataset(
            split="train",
            language=language,
            max_samples=max_train_samples,
        ),
        "validation": CodeSearchNetDataset(
            split="validation",
            language=language,
            max_samples=max_val_samples,
        ),
    }
