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
    
    Use for semantic code search (encode_code) and retrieval queries (encode_query).
    """
    
    config_class = RepoJEPAConfig
    
    def __init__(self, config: RepoJEPAConfig):
        super().__init__(config)
        
        # In the HF model, we store both encoders
        self.context_encoder = RobertaModel.from_pretrained(
            config.base_model,
            add_pooling_layer=False,
        )
        self.target_encoder = RobertaModel.from_pretrained(
            config.base_model,
            add_pooling_layer=False,
        )
        
        # Projection heads
        hidden_size = self.context_encoder.config.hidden_size
        self.context_projector = ProjectionHead(hidden_size, config.hidden_dim)
        self.target_projector = ProjectionHead(hidden_size, config.hidden_dim)
        
        self.post_init()
    
    def encode_code(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode code snippet into embedding space."""
        outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        return self.context_projector(pooled)
    
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode search query (docstring) into embedding space."""
        outputs = self.target_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        return self.target_projector(pooled)

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_hidden / sum_mask
        return hidden_states.mean(dim=1)

    def forward(self, **kwargs):
        # HF requires forward(), we default to code encoding or raise error
        if "input_ids" in kwargs:
            return self.encode_code(kwargs["input_ids"], kwargs.get("attention_mask"))
        raise NotImplementedError("Use .encode_code() or .encode_query() specifically.")


# Register with Auto classes
try:
    from transformers import AutoConfig, AutoModel
    AutoConfig.register("repo-jepa", RepoJEPAConfig)
    AutoModel.register(RepoJEPAConfig, RepoJEPAModel)
except:
    pass

