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
        allow_mock: bool = True,
    ):
        self.split = split
        self.language = language
        self.min_doc_length = min_doc_length
        self.min_code_length = min_code_length
        self.allow_mock = allow_mock
        
        # Load dataset
        self.data = self._load_and_filter(max_samples)
    
    def _load_and_filter(self, max_samples: Optional[int]) -> List[CodeDocPair]:
        """Load and filter CodeSearchNet data."""
        print(f"Loading CodeSearchNet ({self.language}/{self.split})...")
        
        # Try multiple identifiers to be robust
        identifiers = [
            ("claudios/code_search_net", self.language), # Modern parquet version
            ("code_search_net", self.language),         # Traditional version
        ]
        
        dataset = None
        last_error = None
        
        for identifier, subset in identifiers:
            try:
                print(f"Attempting to load: {identifier} ({subset})...")
                # Try loading with streaming to be faster and bypass some disk issues
                dataset = load_dataset(
                    identifier,
                    subset,
                    split=self.split,
                    streaming=True # Use streaming to check if it works without full download
                )
                # Verify we can actually get at least one item
                next(iter(dataset))
                print(f"Successfully connected to: {identifier}")
                break
            except Exception as e:
                print(f"Failed to load {identifier}: {e}")
                last_error = e
                dataset = None
        
        if dataset is None:
            if not self.allow_mock:
                print("ABORTING: Real data loading failed and allow_mock=False")
                raise RuntimeError(f"Could not load real CodeSearchNet dataset. Last error: {last_error}")
                
            print("Falling back to mock data for testing...")
            return self._create_mock_data(max_samples or 10000)
        
        pairs = []
        count = 0
        # Iterate over streaming dataset
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
            
            count += 1
            if max_samples and count >= max_samples:
                break
        
        print(f"Loaded {len(pairs)} code-doc pairs")
        return pairs
    
    def _create_mock_data(self, n_samples: int) -> List[CodeDocPair]:
        """Create mock data for testing without internet."""
        mock_functions = [
            ("def add(a, b):\n    return a + b", "Add two numbers together and return the result."),
            ("def multiply(x, y):\n    return x * y", "Multiply two values and return their product."),
            ("def read_file(path):\n    with open(path) as f:\n        return f.read()", "Read and return the contents of a file from disk."),
            ("def write_json(data, path):\n    import json\n    with open(path, 'w') as f:\n        json.dump(data, f)", "Write data to a JSON file at the specified path."),
            ("def fetch_url(url):\n    import requests\n    return requests.get(url).text", "Fetch and return content from a URL using HTTP GET."),
            ("def sort_list(items):\n    return sorted(items)", "Sort a list in ascending order and return it."),
            ("def filter_none(items):\n    return [x for x in items if x is not None]", "Remove None values from list and return filtered list."),
            ("def calculate_mean(numbers):\n    return sum(numbers) / len(numbers)", "Calculate and return the arithmetic mean of numbers."),
            ("def validate_email(email):\n    import re\n    return bool(re.match(r'^[\\w.-]+@[\\w.-]+\\.\\w+$', email))", "Validate email format using regex pattern matching."),
            ("def hash_password(password):\n    import hashlib\n    return hashlib.sha256(password.encode()).hexdigest()", "Hash password using SHA256 and return hex digest."),
            ("def connect_db(host, port):\n    import sqlite3\n    return sqlite3.connect(f'{host}:{port}')", "Connect to database at specified host and port."),
            ("def parse_json(text):\n    import json\n    return json.loads(text)", "Parse JSON string and return Python object."),
            ("def format_date(dt):\n    return dt.strftime('%Y-%m-%d')", "Format datetime object as ISO date string."),
            ("def split_text(text, sep):\n    return text.split(sep)", "Split text by separator and return list of parts."),
            ("def join_paths(*paths):\n    import os\n    return os.path.join(*paths)", "Join path components into single path string."),
            ("def get_env(key, default=None):\n    import os\n    return os.environ.get(key, default)", "Get environment variable with optional default value."),
            ("def count_words(text):\n    return len(text.split())", "Count and return number of words in text."),
            ("def reverse_string(s):\n    return s[::-1]", "Reverse a string and return the reversed version."),
            ("def find_max(numbers):\n    return max(numbers)", "Find and return the maximum value in a list."),
            ("def find_min(numbers):\n    return min(numbers)", "Find and return the minimum value in a list."),
        ]
        
        pairs = []
        for i in range(n_samples):
            code, doc = mock_functions[i % len(mock_functions)]
            # Add some variation to make embeddings more interesting
            variation = f"  # Variation {i}"
            pairs.append(CodeDocPair(
                code=code + variation,
                docstring=doc + f" (sample {i})",
                func_name=f"func_{i}",
                language="python",
            ))
        
        print(f"Created {len(pairs)} mock code-doc pairs")
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
