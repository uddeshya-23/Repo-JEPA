"""
Repo-JEPA Model Architecture

Dual-encoder JEPA for semantic code search.
Uses pretrained CodeBERT as the backbone with EMA target encoder.
"""

import os
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import RepoJEPAConfig

# Get HF token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")


class RepoJEPA(nn.Module):
    """
    Repo-JEPA: Joint Embedding Predictive Architecture for Code Search.
    
    Architecture:
        - Context Encoder: Encodes code snippets (trainable)
        - Target Encoder: Encodes docstrings/queries (EMA-updated, frozen during forward)
        - Projection Head: Maps encoder outputs to embedding space
    
    Training Objective:
        Predict the target embedding (docstring) from the context embedding (code)
        using VICReg loss to prevent representation collapse.
    """
    
    def __init__(self, config: RepoJEPAConfig):
        super().__init__()
        self.config = config
        
        # Initialize encoders from pretrained CodeBERT
        if config.pretrained_encoder:
            self.context_encoder = RobertaModel.from_pretrained(
                config.pretrained_encoder,
                add_pooling_layer=False,
                token=HF_TOKEN,
                use_safetensors=True
            )

            self.target_encoder = RobertaModel.from_pretrained(
                config.pretrained_encoder,
                add_pooling_layer=False,
                token=HF_TOKEN,
                use_safetensors=True
            )

        else:
            # Random initialization (for testing)
            roberta_config = RobertaConfig(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_dim,
                num_hidden_layers=config.num_encoder_layers,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_dim,
                hidden_dropout_prob=config.hidden_dropout_prob,
                attention_probs_dropout_prob=config.attention_dropout_prob,
                max_position_embeddings=config.max_seq_len + 2,
            )
            self.context_encoder = RobertaModel(roberta_config, add_pooling_layer=False)
            self.target_encoder = RobertaModel(roberta_config, add_pooling_layer=False)
        
        # Freeze target encoder (updated via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Projection heads (map to embedding space)
        hidden_dim = self.context_encoder.config.hidden_size
        self.context_projector = ProjectionHead(hidden_dim, config.hidden_dim)
        self.target_projector = ProjectionHead(hidden_dim, config.hidden_dim)
        
        # Freeze target projector
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        # EMA decay scheduler
        self.ema_decay = config.ema_decay
        self.ema_decay_end = config.ema_decay_end
        
        # Gradient checkpointing
        if config.use_gradient_checkpointing:
            self.context_encoder.gradient_checkpointing_enable()
    
    def forward(
        self,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            code_input_ids: Tokenized code snippets [B, seq_len]
            code_attention_mask: Attention mask for code [B, seq_len]
            doc_input_ids: Tokenized docstrings [B, seq_len]
            doc_attention_mask: Attention mask for docstrings [B, seq_len]
        
        Returns:
            context_embed: Code embeddings [B, hidden_dim]
            target_embed: Docstring embeddings [B, hidden_dim] (detached)
        """
        # Encode code (context) - trainable
        context_output = self.context_encoder(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask,
        ).last_hidden_state  # [B, seq_len, hidden]
        
        # Encode docstring (target) - frozen
        with torch.no_grad():
            target_output = self.target_encoder(
                input_ids=doc_input_ids,
                attention_mask=doc_attention_mask,
            ).last_hidden_state  # [B, seq_len, hidden]
        
        # Pool to single vector (mean pooling over non-padded tokens)
        context_pooled = self._mean_pool(context_output, code_attention_mask)
        target_pooled = self._mean_pool(target_output, doc_attention_mask)
        
        # Project to embedding space
        context_embed = self.context_projector(context_pooled)
        
        with torch.no_grad():
            target_embed = self.target_projector(target_pooled)
        
        return context_embed, target_embed
    
    def encode_code(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode code snippet for inference."""
        output = self.context_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pooled = self._mean_pool(output, attention_mask)
        return self.context_projector(pooled)
    
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode query/docstring for inference."""
        output = self.target_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        pooled = self._mean_pool(output, attention_mask)
        return self.target_projector(pooled)
    
    def _mean_pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over non-padded tokens."""
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        return sum_hidden / sum_mask
    
    @torch.no_grad()
    def update_target_encoder(self, progress: float = 0.0):
        """
        Update target encoder via Exponential Moving Average.
        
        Args:
            progress: Training progress [0, 1] for decay scheduling
        """
        # Linear decay schedule: ema_decay -> ema_decay_end
        decay = self.ema_decay + progress * (self.ema_decay_end - self.ema_decay)
        
        # Update target encoder
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = decay * param_k.data + (1 - decay) * param_q.data
        
        # Update target projector
        for param_q, param_k in zip(
            self.context_projector.parameters(),
            self.target_projector.parameters()
        ):
            param_k.data = decay * param_k.data + (1 - decay) * param_q.data


class ProjectionHead(nn.Module):
    """MLP projection head for embedding space mapping."""
    
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


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Count all parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())
