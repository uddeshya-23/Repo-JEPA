"""
Tests for data loading and collation.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CodeSearchNetDataset, CodeDocCollator, get_tokenizer


class TestCodeSearchNetDataset:
    """Tests for CodeSearchNet dataset."""
    
    def test_mock_data_creation(self):
        """Test dataset creates mock data for testing."""
        dataset = CodeSearchNetDataset(
            split="train",
            max_samples=50,
        )
        
        assert len(dataset) > 0
    
    def test_getitem_returns_dict(self):
        """Test __getitem__ returns expected dict."""
        dataset = CodeSearchNetDataset(
            split="train",
            max_samples=10,
        )
        
        item = dataset[0]
        
        assert "code" in item
        assert "docstring" in item
        assert "func_name" in item
    
    def test_code_and_doc_are_strings(self):
        """Test code and docstring are strings."""
        dataset = CodeSearchNetDataset(
            split="train",
            max_samples=10,
        )
        
        item = dataset[0]
        
        assert isinstance(item["code"], str)
        assert isinstance(item["docstring"], str)
        assert len(item["code"]) > 0
        assert len(item["docstring"]) > 0


class TestCodeDocCollator:
    """Tests for data collator."""
    
    @pytest.fixture
    def tokenizer(self):
        """Get tokenizer (may need to download)."""
        try:
            return get_tokenizer("microsoft/codebert-base")
        except Exception:
            # Fallback for offline testing
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("roberta-base")
    
    @pytest.fixture
    def collator(self, tokenizer):
        return CodeDocCollator(
            tokenizer=tokenizer,
            max_code_length=64,
            max_doc_length=32,
        )
    
    def test_collate_returns_tensors(self, collator):
        """Test collator returns PyTorch tensors."""
        batch = [
            {"code": "def foo(): pass", "docstring": "A function."},
            {"code": "def bar(x): return x", "docstring": "Identity function."},
        ]
        
        output = collator(batch)
        
        assert isinstance(output["code_input_ids"], torch.Tensor)
        assert isinstance(output["code_attention_mask"], torch.Tensor)
        assert isinstance(output["doc_input_ids"], torch.Tensor)
        assert isinstance(output["doc_attention_mask"], torch.Tensor)
    
    def test_output_shapes(self, collator):
        """Test output tensor shapes."""
        batch = [
            {"code": "def foo(): pass", "docstring": "A function."},
            {"code": "def bar(x): return x", "docstring": "Identity function."},
        ]
        
        output = collator(batch)
        
        assert output["code_input_ids"].shape == (2, 64)  # batch, max_code_length
        assert output["doc_input_ids"].shape == (2, 32)   # batch, max_doc_length
    
    def test_attention_mask_valid(self, collator):
        """Test attention mask has valid values."""
        batch = [
            {"code": "def foo(): pass", "docstring": "A function."},
        ]
        
        output = collator(batch)
        
        # Should be 0 or 1
        assert output["code_attention_mask"].min() >= 0
        assert output["code_attention_mask"].max() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
