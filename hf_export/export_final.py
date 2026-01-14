
import os
import torch
from safetensors.torch import save_file
from modeling_repo_jepa import RepoJEPAConfig, RepoJEPAModel

def export_checkpoint(checkpoint_path, output_dir="hf_export"):
    """
    Convert a Repo-JEPA .pt checkpoint to Hugging Face safetensors format.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract config
    config_dict = checkpoint.get("config", {})
    # Map REPO_JEPA_BASE keys to RepoJEPAConfig keys if necessary
    # (Assuming they match based on our code)
    config = RepoJEPAConfig(**config_dict)
    
    # Initialize the HF-compatible model
    print("Initializing HF-compatible model...")
    hf_model = RepoJEPAModel(config)
    
    # Load state dict
    # The checkpoint has separate encoders: context_encoder, target_encoder, etc.
    # Our hf_model structure matches this.
    state_dict = checkpoint["model_state_dict"]
    
    # Check for missing/extra keys
    hf_state_dict = hf_model.state_dict()
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k in hf_state_dict:
            new_state_dict[k] = v
        else:
            print(f"Skipping extra key: {k}")
            
    hf_model.load_state_dict(new_state_dict, strict=False)
    
    # Save as safetensors
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving weights to {weights_path}...")
    save_file(hf_model.state_dict(), weights_path)
    
    # Save config
    config.save_pretrained(output_dir)
    print("✓ Export complete!")
    print(f"\nNow you can upload the '{output_dir}' directory to Hugging Face.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to .pt checkpoint")
    parser.add_argument("--out", type=str, default="hf_export", help="Output directory")
    args = parser.parse_args()
    
    export_checkpoint(args.checkpoint, args.out)
