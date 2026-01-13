"""
Data Collator for Repo-JEPA

Tokenizes and batches code-docstring pairs for training.
"""

import os
from typing import List, Dict, Any

import torch
from transformers import PreTrainedTokenizer, AutoTokenizer

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use os.environ directly


class CodeDocCollator:
    """
    Collator for code-docstring pairs.
    
    Tokenizes both code and docstring, pads to max_length,
    and returns tensors ready for the model.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_code_length: int = 256,
        max_doc_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_doc_length = max_doc_length
    
    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of code-docstring pairs.
        
        Args:
            batch: List of dicts with 'code' and 'docstring' keys
        
        Returns:
            Dict with tokenized tensors:
                - code_input_ids, code_attention_mask
                - doc_input_ids, doc_attention_mask
        """
        codes = [item["code"] for item in batch]
        docstrings = [item["docstring"] for item in batch]
        
        # Tokenize code
        code_encoded = self.tokenizer(
            codes,
            padding="max_length",
            truncation=True,
            max_length=self.max_code_length,
            return_tensors="pt",
        )
        
        # Tokenize docstrings
        doc_encoded = self.tokenizer(
            docstrings,
            padding="max_length",
            truncation=True,
            max_length=self.max_doc_length,
            return_tensors="pt",
        )
        
        return {
            "code_input_ids": code_encoded["input_ids"],
            "code_attention_mask": code_encoded["attention_mask"],
            "doc_input_ids": doc_encoded["input_ids"],
            "doc_attention_mask": doc_encoded["attention_mask"],
        }


def get_tokenizer(model_name: str = "microsoft/codebert-base") -> PreTrainedTokenizer:
    """Load tokenizer for CodeBERT with fallback."""
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except Exception as e:
        print(f"Could not load {model_name} tokenizer: {e}")
        print("Falling back to roberta-base tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base", token=hf_token)
    return tokenizer
