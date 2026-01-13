"""
Code Search Evaluation (MRR Benchmark)

Evaluates Repo-JEPA on CodeSearchNet using Mean Reciprocal Rank.
"""

import argparse
from typing import List, Dict, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..config import RepoJEPAConfig
from ..model import RepoJEPA
from ..data import CodeSearchNetDataset, CodeDocCollator, get_tokenizer


def mean_reciprocal_rank(rankings: List[int]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        rankings: List of ranks (1-indexed) of correct items
    
    Returns:
        MRR score between 0 and 1
    """
    if not rankings:
        return 0.0
    
    reciprocals = [1.0 / r for r in rankings if r > 0]
    return sum(reciprocals) / len(rankings)


@torch.no_grad()
def evaluate_code_search(
    model: RepoJEPA,
    dataset: CodeSearchNetDataset,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_eval_samples: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate model on code search task.
    
    For each query (docstring), rank all code snippets.
    Compute MRR based on rank of the correct code.
    
    Args:
        model: Trained Repo-JEPA model
        dataset: Evaluation dataset
        tokenizer: Tokenizer for encoding
        device: Device to run evaluation on
        batch_size: Batch size for encoding
        max_eval_samples: Maximum samples to evaluate
    
    Returns:
        Dict with MRR and other metrics
    """
    model.eval()
    
    # Limit samples
    n_samples = min(len(dataset), max_eval_samples)
    
    # Encode all code and docstrings
    collator = CodeDocCollator(tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    
    all_code_embeds = []
    all_doc_embeds = []
    
    print("Encoding samples...")
    for batch in tqdm(loader, desc="Encoding"):
        code_input_ids = batch["code_input_ids"].to(device)
        code_attention_mask = batch["code_attention_mask"].to(device)
        doc_input_ids = batch["doc_input_ids"].to(device)
        doc_attention_mask = batch["doc_attention_mask"].to(device)
        
        # Encode
        code_embed = model.encode_code(code_input_ids, code_attention_mask)
        doc_embed = model.encode_query(doc_input_ids, doc_attention_mask)
        
        all_code_embeds.append(code_embed.cpu())
        all_doc_embeds.append(doc_embed.cpu())
        
        if len(all_code_embeds) * batch_size >= max_eval_samples:
            break
    
    # Concatenate
    code_embeds = torch.cat(all_code_embeds, dim=0)[:n_samples]
    doc_embeds = torch.cat(all_doc_embeds, dim=0)[:n_samples]
    
    # Normalize for cosine similarity
    code_embeds = F.normalize(code_embeds, dim=-1)
    doc_embeds = F.normalize(doc_embeds, dim=-1)
    
    # Compute similarity matrix [n_docs, n_codes]
    print("Computing similarity matrix...")
    similarity = doc_embeds @ code_embeds.T  # [N, N]
    
    # For each doc, find rank of correct code (diagonal)
    # Correct code for doc[i] is code[i]
    rankings = []
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    
    for i in range(n_samples):
        # Sort by similarity (descending)
        sorted_indices = torch.argsort(similarity[i], descending=True)
        
        # Find rank of correct code (1-indexed)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        rankings.append(rank)
        
        if rank == 1:
            hits_at_1 += 1
        if rank <= 5:
            hits_at_5 += 1
        if rank <= 10:
            hits_at_10 += 1
    
    mrr = mean_reciprocal_rank(rankings)
    
    results = {
        "mrr": mrr,
        "hits@1": hits_at_1 / n_samples,
        "hits@5": hits_at_5 / n_samples,
        "hits@10": hits_at_10 / n_samples,
        "n_samples": n_samples,
        "avg_rank": np.mean(rankings),
        "median_rank": np.median(rankings),
        "similarity": similarity.tolist(), # Convert to list for JSON serialization
        "rankings": rankings,
    }

    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Repo-JEPA on CodeSearchNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="codesearchnet", help="Dataset to evaluate on")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max eval samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-results", type=str, help="Path to save evaluation results (JSON)")

    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = RepoJEPAConfig(**checkpoint.get("config", {}))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    model = RepoJEPA(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load tokenizer and dataset
    tokenizer = get_tokenizer(config.pretrained_encoder)
    dataset = CodeSearchNetDataset(split="test", language="python", max_samples=args.max_samples)
    
    # Evaluate
    results = evaluate_code_search(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size,
        max_eval_samples=args.max_samples,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"MRR:        {results['mrr']:.4f}")
    print(f"Hits@1:     {results['hits@1']:.4f}")
    print(f"Hits@5:     {results['hits@5']:.4f}")
    print(f"Hits@10:    {results['hits@10']:.4f}")
    print(f"Avg Rank:   {results['avg_rank']:.2f}")
    print(f"Median Rank: {results['median_rank']:.2f}")
    print(f"N Samples:  {results['n_samples']}")
    print("=" * 50)
    
    # Success criteria
    if results['mrr'] >= 0.6:
        print("✅ SUCCESS: MRR >= 0.6 (target achieved)")
    else:
        print(f"⚠️  MRR below target (0.6). Current: {results['mrr']:.4f}")
        
    # Save results
    if args.save_results:
        import json
        with open(args.save_results, "w") as f:
            json.dump(results, f)
        print(f"💾 Results saved to {args.save_results}")



if __name__ == "__main__":
    main()
