"""
Repo-JEPA Training Loop

Main training script with gradient accumulation, mixed precision,
and W&B logging.
"""

import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .config import RepoJEPAConfig, REPO_JEPA_BASE
from .model import RepoJEPA, count_parameters, count_all_parameters
from .loss import VICRegLoss, CombinedLoss
from .data import CodeSearchNetDataset, CodeDocCollator, get_tokenizer


def train(
    config: RepoJEPAConfig,
    output_dir: str = "checkpoints",
    use_wandb: bool = True,
    sanity_check: bool = False,
):
    """
    Main training function.
    
    Args:
        config: Model and training configuration
        output_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
        sanity_check: If True, run a quick sanity check (10 steps)
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(config.pretrained_encoder)
    
    # Load dataset
    print("Loading dataset...")
    max_samples = 100 if sanity_check else None
    train_dataset = CodeSearchNetDataset(
        split="train",
        language="python",
        max_samples=max_samples,
    )
    
    # Data loader
    collator = CodeDocCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Initialize model
    print("Initializing model...")
    model = RepoJEPA(config).to(device)
    
    # Print parameter count
    trainable_params = count_parameters(model)
    total_params = count_all_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Initialize loss
    loss_fn = VICRegLoss(
        lambda_mse=config.lambda_mse,
        lambda_var=config.lambda_var,
        lambda_cov=config.lambda_cov,
        variance_threshold=config.variance_threshold,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.max_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="cos",
    )
    
    # Mixed precision
    scaler = GradScaler() if config.use_fp16 and device.type == "cuda" else None
    
    # W&B initialization
    if use_wandb and WANDB_AVAILABLE and not sanity_check:
        wandb.init(
            project="repo-jepa",
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(config),
        )
    
    # Training loop
    print(f"\nStarting training for {config.max_epochs} epochs...")
    print(f"Effective batch size: {config.effective_batch_size}")
    
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics: Dict[str, float] = {}
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.max_epochs}",
            leave=True,
        )
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            code_input_ids = batch["code_input_ids"].to(device)
            code_attention_mask = batch["code_attention_mask"].to(device)
            doc_input_ids = batch["doc_input_ids"].to(device)
            doc_attention_mask = batch["doc_attention_mask"].to(device)
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    context_embed, target_embed = model(
                        code_input_ids,
                        code_attention_mask,
                        doc_input_ids,
                        doc_attention_mask,
                    )
                    loss, loss_dict = loss_fn(context_embed, target_embed.detach())
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                context_embed, target_embed = model(
                    code_input_ids,
                    code_attention_mask,
                    doc_input_ids,
                    doc_attention_mask,
                )
                loss, loss_dict = loss_fn(context_embed, target_embed.detach())
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                scheduler.step()
                
                # Update target encoder (EMA)
                progress = global_step / total_steps if total_steps > 0 else 0
                model.update_target_encoder(progress)
                
                global_step += 1
                
                # Logging
                epoch_loss += loss_dict["loss/total"]
                for k, v in loss_dict.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss_dict['loss/total']:.4f}",
                    "var": f"{loss_dict.get('metrics/pred_std', 0):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
                
                # Collapse detection
                if loss_dict.get("metrics/pred_std", 1) < 0.01:
                    print("\n⚠️  WARNING: Embedding variance collapsed! Adjusting VICReg weights...")
                
                # W&B logging
                if use_wandb and WANDB_AVAILABLE and not sanity_check and global_step % 10 == 0:
                    wandb.log({
                        **loss_dict,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                    })
            
            # Sanity check: stop early
            if sanity_check and global_step >= 10:
                print("\n✅ Sanity check passed! Model can forward and backward.")
                return
        
        # Epoch summary
        avg_loss = epoch_loss / max(1, global_step)
        print(f"\nEpoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = output_path / "best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": vars(config),
            }, checkpoint_path)
            print(f"💾 Saved best checkpoint to {checkpoint_path}")
    
    # Final save
    final_path = output_path / "final.pt"
    torch.save({
        "epoch": config.max_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": avg_loss,
        "config": vars(config),
    }, final_path)
    print(f"\n🎉 Training complete! Final model saved to {final_path}")
    
    if use_wandb and WANDB_AVAILABLE and not sanity_check:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Repo-JEPA")
    parser.add_argument("--sanity-check", action="store_true", help="Run quick sanity check")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained encoder")
    
    args = parser.parse_args()
    
    # Configure model
    config = REPO_JEPA_BASE
    config.max_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    if args.no_pretrained:
        config.pretrained_encoder = None
    
    # Run training
    train(
        config=config,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        sanity_check=args.sanity_check,
    )


if __name__ == "__main__":
    main()
