"""
Tests for Repo-JEPA model architecture.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RepoJEPAConfig, REPO_JEPA_SMALL
from src.model import RepoJEPA, ProjectionHead, count_parameters


class TestProjectionHead:
    """Tests for projection head."""
    
    def test_forward_shape(self):
        """Test output shape."""
        head = ProjectionHead(768, 256)
        x = torch.randn(8, 768)
        out = head(x)
        assert out.shape == (8, 256)
    
    def test_batch_norm_training(self):
        """Test batch norm works in training mode."""
        head = ProjectionHead(128, 64)
        head.train()
        x = torch.randn(16, 128)
        out = head(x)
        assert out.shape == (16, 64)


class TestRepoJEPA:
    """Tests for Repo-JEPA model."""
    
    @pytest.fixture
    def config(self):
        """Small config for testing."""
        return RepoJEPAConfig(
            hidden_dim=128,
            num_encoder_layers=2,
            num_attention_heads=2,
            intermediate_dim=256,
            vocab_size=1000,
            max_seq_len=64,
            pretrained_encoder=None,  # Random init for testing
        )
    
    @pytest.fixture
    def model(self, config):
        """Create model for testing."""
        return RepoJEPA(config)
    
    def test_forward_shape(self, model, config):
        """Test forward pass output shapes."""
        batch_size = 4
        seq_len = 32
        
        code_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        doc_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        doc_mask = torch.ones(batch_size, seq_len)
        
        context_embed, target_embed = model(
            code_ids, code_mask, doc_ids, doc_mask
        )
        
        assert context_embed.shape == (batch_size, config.hidden_dim)
        assert target_embed.shape == (batch_size, config.hidden_dim)
    
    def test_encode_code(self, model, config):
        """Test code encoding."""
        batch_size = 4
        seq_len = 32
        
        code_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        code_mask = torch.ones(batch_size, seq_len)
        
        embed = model.encode_code(code_ids, code_mask)
        assert embed.shape == (batch_size, config.hidden_dim)
    
    def test_encode_query(self, model, config):
        """Test query encoding."""
        batch_size = 4
        seq_len = 32
        
        doc_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        doc_mask = torch.ones(batch_size, seq_len)
        
        embed = model.encode_query(doc_ids, doc_mask)
        assert embed.shape == (batch_size, config.hidden_dim)
    
    def test_target_encoder_frozen(self, model):
        """Test that target encoder is frozen."""
        for param in model.target_encoder.parameters():
            assert not param.requires_grad
    
    def test_ema_update(self, model, config):
        """Test EMA update changes target encoder."""
        # Get initial target weights
        initial_weight = model.target_encoder.embeddings.word_embeddings.weight.clone()
        
        # Modify context encoder
        with torch.no_grad():
            model.context_encoder.embeddings.word_embeddings.weight += 1.0
        
        # Update target via EMA
        model.update_target_encoder(progress=0.5)
        
        # Check target changed
        new_weight = model.target_encoder.embeddings.word_embeddings.weight
        assert not torch.allclose(initial_weight, new_weight)
    
    def test_parameter_count(self, model):
        """Test parameter counting."""
        trainable = count_parameters(model)
        assert trainable > 0


class TestConfig:
    """Tests for configuration."""
    
    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = RepoJEPAConfig(batch_size=32, gradient_accumulation_steps=16)
        assert config.effective_batch_size == 512
    
    def test_parameter_estimate(self):
        """Test parameter estimate."""
        config = REPO_JEPA_SMALL
        estimate = config.num_parameters_estimate
        assert "M" in estimate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
