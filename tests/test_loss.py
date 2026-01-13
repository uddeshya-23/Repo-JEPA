"""
Tests for VICReg loss functions.
"""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loss import VICRegLoss, InfoNCELoss, CombinedLoss


class TestVICRegLoss:
    """Tests for VICReg loss."""
    
    @pytest.fixture
    def loss_fn(self):
        return VICRegLoss(
            lambda_mse=1.0,
            lambda_var=25.0,
            lambda_cov=1.0,
        )
    
    def test_forward_returns_scalar(self, loss_fn):
        """Test loss returns scalar."""
        pred = torch.randn(32, 128)
        target = torch.randn(32, 128)
        
        loss, loss_dict = loss_fn(pred, target)
        
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_loss_dict_keys(self, loss_fn):
        """Test loss dict has expected keys."""
        pred = torch.randn(32, 128)
        target = torch.randn(32, 128)
        
        _, loss_dict = loss_fn(pred, target)
        
        assert "loss/total" in loss_dict
        assert "loss/invariance" in loss_dict
        assert "loss/variance" in loss_dict
        assert "loss/covariance" in loss_dict
        assert "metrics/pred_std" in loss_dict
    
    def test_identical_inputs_low_invariance(self, loss_fn):
        """Test identical inputs have low invariance loss."""
        embeddings = torch.randn(32, 128)
        
        _, loss_dict = loss_fn(embeddings, embeddings)
        
        assert loss_dict["loss/invariance"] < 0.01
    
    def test_collapsed_embeddings_high_variance_loss(self, loss_fn):
        """Test collapsed embeddings have high variance loss."""
        # All embeddings are identical (collapsed)
        collapsed = torch.ones(32, 128)
        target = torch.randn(32, 128)
        
        _, loss_dict = loss_fn(collapsed, target)
        
        # Variance loss should be high when std is near 0
        assert loss_dict["loss/variance"] > 0.5
    
    def test_diverse_embeddings_low_variance_loss(self, loss_fn):
        """Test diverse embeddings have low variance loss."""
        # High variance embeddings
        diverse = torch.randn(32, 128) * 5.0
        target = torch.randn(32, 128) * 5.0
        
        _, loss_dict = loss_fn(diverse, target)
        
        # Variance loss should be low
        assert loss_dict["loss/variance"] < 0.1
    
    def test_correlated_dims_high_cov_loss(self, loss_fn):
        """Test correlated dimensions have high covariance loss."""
        # Create embeddings with correlated dimensions
        base = torch.randn(32, 1).repeat(1, 128)
        noise = torch.randn(32, 128) * 0.01
        correlated = base + noise
        target = torch.randn(32, 128)
        
        _, loss_dict = loss_fn(correlated, target)
        
        # Covariance loss should be higher than for uncorrelated
        assert loss_dict["loss/covariance"] > 0.1


class TestInfoNCELoss:
    """Tests for InfoNCE contrastive loss."""
    
    @pytest.fixture
    def loss_fn(self):
        return InfoNCELoss(temperature=0.07)
    
    def test_forward_returns_scalar(self, loss_fn):
        """Test loss returns scalar."""
        code_embed = torch.randn(32, 128)
        doc_embed = torch.randn(32, 128)
        
        loss, loss_dict = loss_fn(code_embed, doc_embed)
        
        assert loss.shape == ()
        assert loss.item() > 0
    
    def test_identical_pairs_high_accuracy(self, loss_fn):
        """Test identical pairs give high accuracy."""
        embeddings = torch.randn(32, 128)
        
        _, loss_dict = loss_fn(embeddings, embeddings)
        
        # With identical embeddings, accuracy should be 1.0
        assert loss_dict["metrics/acc_c2d"] == 1.0
        assert loss_dict["metrics/acc_d2c"] == 1.0
    
    def test_random_pairs_low_accuracy(self, loss_fn):
        """Test random pairs have low accuracy initially."""
        code_embed = torch.randn(32, 128)
        doc_embed = torch.randn(32, 128)
        
        _, loss_dict = loss_fn(code_embed, doc_embed)
        
        # Random chance is 1/32 ≈ 0.03
        # Allow some variance
        assert loss_dict["metrics/acc_c2d"] < 0.5


class TestCombinedLoss:
    """Tests for combined VICReg + InfoNCE loss."""
    
    def test_forward_combines_losses(self):
        """Test combined loss sums both."""
        loss_fn = CombinedLoss(vicreg_weight=1.0, infonce_weight=1.0)
        
        pred = torch.randn(32, 128)
        target = torch.randn(32, 128)
        
        total_loss, loss_dict = loss_fn(pred, target)
        
        assert "loss/vicreg" in loss_dict
        assert "loss/infonce" in loss_dict
        assert total_loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
