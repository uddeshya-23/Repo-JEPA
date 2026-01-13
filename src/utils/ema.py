"""
Exponential Moving Average (EMA) Updater

Updates target network parameters as EMA of source network.
"""

import torch
import torch.nn as nn
from typing import Optional


class EMAUpdater:
    """
    Exponential Moving Average updater for JEPA target encoder.
    
    target_params = decay * target_params + (1 - decay) * source_params
    """
    
    def __init__(
        self,
        source_model: nn.Module,
        target_model: nn.Module,
        base_decay: float = 0.996,
        final_decay: float = 1.0,
        total_steps: Optional[int] = None,
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.base_decay = base_decay
        self.final_decay = final_decay
        self.total_steps = total_steps
        self.current_step = 0
    
    @property
    def decay(self) -> float:
        """Get current decay value (with optional scheduling)."""
        if self.total_steps is None:
            return self.base_decay
        
        progress = min(1.0, self.current_step / self.total_steps)
        return self.base_decay + progress * (self.final_decay - self.base_decay)
    
    @torch.no_grad()
    def update(self) -> float:
        """
        Perform one EMA update step.
        
        Returns:
            Current decay value used
        """
        decay = self.decay
        
        for source_param, target_param in zip(
            self.source_model.parameters(),
            self.target_model.parameters()
        ):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)
        
        self.current_step += 1
        return decay
    
    def copy_params(self):
        """Copy source parameters to target (for initialization)."""
        for source_param, target_param in zip(
            self.source_model.parameters(),
            self.target_model.parameters()
        ):
            target_param.data.copy_(source_param.data)
