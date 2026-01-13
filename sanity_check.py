"""
Quick sanity check for Repo-JEPA.
Run: python sanity_check.py
"""

import os
import sys

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import REPO_JEPA_SMALL
from src.model import RepoJEPA, count_parameters, count_all_parameters
from src.loss import VICRegLoss


def main():
    print("=" * 50)
    print("Repo-JEPA Sanity Check")
    print("=" * 50)
    
    # Use small config with random init for quick test
    config = REPO_JEPA_SMALL
    config.pretrained_encoder = None  # Random init for speed
    
    print("\n1. Creating model...")
    model = RepoJEPA(config)
    
    trainable = count_parameters(model)
    total = count_all_parameters(model)
    print(f"   Trainable params: {trainable:,}")
    print(f"   Total params: {total:,}")
    
    print("\n2. Creating dummy input...")
    batch_size = 4
    seq_len = 32
    
    code_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    code_mask = torch.ones(batch_size, seq_len)
    doc_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    doc_mask = torch.ones(batch_size, seq_len)
    
    print(f"   Input shape: {code_ids.shape}")
    
    print("\n3. Forward pass...")
    context_embed, target_embed = model(code_ids, code_mask, doc_ids, doc_mask)
    print(f"   Context embedding: {context_embed.shape}")
    print(f"   Target embedding: {target_embed.shape}")
    
    print("\n4. Computing VICReg loss...")
    loss_fn = VICRegLoss()
    loss, loss_dict = loss_fn(context_embed, target_embed.detach())
    
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Invariance: {loss_dict['loss/invariance']:.4f}")
    print(f"   Variance: {loss_dict['loss/variance']:.4f}")
    print(f"   Covariance: {loss_dict['loss/covariance']:.4f}")
    print(f"   Pred std: {loss_dict['metrics/pred_std']:.4f}")
    
    print("\n5. Backward pass...")
    loss.backward()
    print("   Gradients computed successfully!")
    
    print("\n6. EMA update...")
    model.update_target_encoder(progress=0.5)
    print("   Target encoder updated!")
    
    print("\n" + "=" * 50)
    print("✅ SUCCESS: All components working correctly!")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
