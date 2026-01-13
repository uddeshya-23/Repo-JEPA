"""
Hugging Face Export for Repo-JEPA

This file enables loading Repo-JEPA with AutoModel.from_pretrained()
using trust_remote_code=True.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, RobertaModel


class RepoJEPAConfig(PretrainedConfig):
    """Configuration for Repo-JEPA model."""
    
    model_type = "repo-jepa"
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_encoder_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_dim: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        vocab_size: int = 50265,
        max_seq_len: int = 512,
        pad_token_id: int = 1,
        base_model: str = "microsoft/codebert-base",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.base_model = base_model


class ProjectionHead(nn.Module):
    """MLP projection head."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class RepoJEPAModel(PreTrainedModel):
    """
    Repo-JEPA: Joint Embedding Predictive Architecture for Code Search.
    
    Use for semantic code search and retrieval tasks.
    """
    
    config_class = RepoJEPAConfig
    
    def __init__(self, config: RepoJEPAConfig):
        super().__init__(config)
        
        # Load base encoder
        self.encoder = RobertaModel.from_pretrained(
            config.base_model,
            add_pooling_layer=False,
        )
        
        # Projection head
        hidden_size = self.encoder.config.hidden_size
        self.projector = ProjectionHead(hidden_size, config.hidden_dim)
        
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Encode input and return embeddings.
        
        Args:
            input_ids: Tokenized input [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
        
        Returns:
            Embeddings [B, hidden_dim]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Mean pooling
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Project
        embeddings = self.projector(pooled)
        
        return embeddings
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for forward - encode text to embedding."""
        return self.forward(input_ids, attention_mask)


# Register with Auto classes
from transformers import AutoConfig, AutoModel

AutoConfig.register("repo-jepa", RepoJEPAConfig)
AutoModel.register(RepoJEPAConfig, RepoJEPAModel)
