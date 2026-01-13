"""
VICReg Loss for Repo-JEPA

Variance-Invariance-Covariance Regularization to prevent embedding collapse.
Reference: https://arxiv.org/abs/2105.04906
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VICRegLoss(nn.Module):
    """
    VICReg Loss: Variance-Invariance-Covariance Regularization.
    
    Components:
        - Invariance (MSE): Predicted embeddings should match targets
        - Variance: Embeddings should have high variance (prevents collapse)
        - Covariance: Embedding dimensions should be decorrelated
    
    Loss = λ_mse * L_invariance + λ_var * L_variance + λ_cov * L_covariance
    """
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        variance_threshold: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.variance_threshold = variance_threshold
        self.eps = eps
    
    def forward(
        self,
        pred_embed: torch.Tensor,
        target_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VICReg loss.
        
        Args:
            pred_embed: Predicted embeddings [B, D]
            target_embed: Target embeddings [B, D] (detached)
        
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Individual loss components for logging
        """
        # Invariance loss (MSE between predictions and targets)
        inv_loss = F.mse_loss(pred_embed, target_embed)
        
        # Variance loss (ensure embedding variance stays high)
        var_loss = self._variance_loss(pred_embed) + self._variance_loss(target_embed)
        var_loss = var_loss / 2  # Average
        
        # Covariance loss (decorrelate embedding dimensions)
        cov_loss = self._covariance_loss(pred_embed) + self._covariance_loss(target_embed)
        cov_loss = cov_loss / 2  # Average
        
        # Combined loss
        total_loss = (
            self.lambda_mse * inv_loss +
            self.lambda_var * var_loss +
            self.lambda_cov * cov_loss
        )
        
        # Logging dict
        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/invariance": inv_loss.item(),
            "loss/variance": var_loss.item(),
            "loss/covariance": cov_loss.item(),
            "metrics/pred_std": pred_embed.std(dim=0).mean().item(),
            "metrics/target_std": target_embed.std(dim=0).mean().item(),
        }
        
        return total_loss, loss_dict
    
    def _variance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Variance regularization: penalize low variance (collapse).
        
        Encourages std(embeddings) >= threshold across batch.
        """
        # Standard deviation across batch for each dimension
        std = torch.sqrt(embeddings.var(dim=0) + self.eps)
        
        # Hinge loss: penalize if std < threshold
        var_loss = F.relu(self.variance_threshold - std).mean()
        
        return var_loss
    
    def _covariance_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Covariance regularization: decorrelate embedding dimensions.
        
        Penalizes off-diagonal elements of covariance matrix.
        """
        batch_size, dim = embeddings.shape
        
        # Center embeddings
        embeddings = embeddings - embeddings.mean(dim=0)
        
        # Compute covariance matrix
        cov = (embeddings.T @ embeddings) / (batch_size - 1)
        
        # Zero out diagonal (we only penalize off-diagonal)
        cov_loss = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        cov_loss = cov_loss / dim  # Normalize by dimension
        
        return cov_loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Contrastive) Loss - Alternative to VICReg.
    
    Useful for retrieval tasks where we want to maximize similarity
    between positive pairs and minimize similarity with negatives.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        code_embed: torch.Tensor,
        doc_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute InfoNCE loss (symmetric).
        
        Assumes code_embed[i] and doc_embed[i] are positive pairs.
        """
        # Normalize embeddings
        code_embed = F.normalize(code_embed, dim=-1)
        doc_embed = F.normalize(doc_embed, dim=-1)
        
        # Compute similarity matrix
        logits = code_embed @ doc_embed.T / self.temperature  # [B, B]
        
        # Labels: diagonal entries are positives
        batch_size = code_embed.shape[0]
        labels = torch.arange(batch_size, device=code_embed.device)
        
        # Symmetric loss
        loss_c2d = F.cross_entropy(logits, labels)
        loss_d2c = F.cross_entropy(logits.T, labels)
        total_loss = (loss_c2d + loss_d2c) / 2
        
        # Compute accuracy for logging
        with torch.no_grad():
            pred_c2d = logits.argmax(dim=1)
            pred_d2c = logits.T.argmax(dim=1)
            acc_c2d = (pred_c2d == labels).float().mean()
            acc_d2c = (pred_d2c == labels).float().mean()
        
        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/code_to_doc": loss_c2d.item(),
            "loss/doc_to_code": loss_d2c.item(),
            "metrics/acc_c2d": acc_c2d.item(),
            "metrics/acc_d2c": acc_d2c.item(),
        }
        
        return total_loss, loss_dict


class CombinedLoss(nn.Module):
    """
    Combined VICReg + InfoNCE loss for robust training.
    
    VICReg prevents collapse, InfoNCE provides retrieval signal.
    """
    
    def __init__(
        self,
        vicreg_weight: float = 1.0,
        infonce_weight: float = 1.0,
        **vicreg_kwargs,
    ):
        super().__init__()
        self.vicreg = VICRegLoss(**vicreg_kwargs)
        self.infonce = InfoNCELoss()
        self.vicreg_weight = vicreg_weight
        self.infonce_weight = infonce_weight
    
    def forward(
        self,
        pred_embed: torch.Tensor,
        target_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        vicreg_loss, vicreg_dict = self.vicreg(pred_embed, target_embed)
        infonce_loss, infonce_dict = self.infonce(pred_embed, target_embed)
        
        total_loss = self.vicreg_weight * vicreg_loss + self.infonce_weight * infonce_loss
        
        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/vicreg": vicreg_loss.item(),
            "loss/infonce": infonce_loss.item(),
            **{f"vicreg/{k}": v for k, v in vicreg_dict.items()},
            **{f"infonce/{k}": v for k, v in infonce_dict.items()},
        }
        
        return total_loss, loss_dict
