"""
Repo-JEPA Model Configuration

Defines hyperparameters for the dual-encoder architecture.
Optimized for ~110M parameters to fit consumer GPUs.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RepoJEPAConfig:
    """Configuration for Repo-JEPA model."""
    
    # Model architecture
    hidden_dim: int = 768
    num_encoder_layers: int = 12
    num_attention_heads: int = 12
    intermediate_dim: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    
    # Vocabulary & tokenization
    vocab_size: int = 50265  # RoBERTa vocab size
    max_seq_len: int = 512
    pad_token_id: int = 1
    
    # JEPA-specific
    ema_decay: float = 0.996  # EMA decay for target encoder
    ema_decay_end: float = 1.0  # Final EMA decay value
    mask_ratio: float = 0.15  # Masking ratio for context
    
    # VICReg loss weights
    lambda_mse: float = 1.0
    lambda_var: float = 25.0
    lambda_cov: float = 1.0
    variance_threshold: float = 1.0  # Minimum std for variance loss
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 16  # Effective batch = 512
    
    # Hardware optimization
    use_fp16: bool = True
    use_gradient_checkpointing: bool = False
    
    # Paths
    pretrained_encoder: Optional[str] = field(
        default="microsoft/codebert-base",
        metadata={"help": "Pretrained encoder to initialize from"}
    )
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def num_parameters_estimate(self) -> str:
        """Rough parameter count estimate."""
        # Embeddings: vocab_size * hidden_dim
        embeddings = self.vocab_size * self.hidden_dim
        # Each transformer layer: 4 * hidden_dim^2 (attention) + 2 * hidden_dim * intermediate_dim (FFN)
        per_layer = 4 * (self.hidden_dim ** 2) + 2 * self.hidden_dim * self.intermediate_dim
        total_layers = self.num_encoder_layers * per_layer
        total = (embeddings + total_layers) * 2  # x2 for dual encoder
        return f"~{total // 1_000_000}M parameters"


# Preset configurations
REPO_JEPA_SMALL = RepoJEPAConfig(
    hidden_dim=384,
    num_encoder_layers=6,
    num_attention_heads=6,
    intermediate_dim=1536,
)

REPO_JEPA_BASE = RepoJEPAConfig()  # Default ~110M params

REPO_JEPA_LARGE = RepoJEPAConfig(
    hidden_dim=1024,
    num_encoder_layers=24,
    num_attention_heads=16,
    intermediate_dim=4096,
)
