
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def plot_evaluation(results_path: str, output_dir: str = "plots"):
    """
    Generate plots from evaluation results.
    """
    with open(results_path, "r") as f:
        results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Similarity Matrix Distribution
    plt.figure(figsize=(10, 6))
    similarity = np.array(results["similarity"])
    
    # Diagonal (correct matches) vs Off-diagonal (incorrect matches)
    diag = np.diag(similarity)
    off_diag = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    
    sns.histplot(diag, color="green", label="Correct Pairs (Diagonal)", kde=True, stat="density", alpha=0.5)
    sns.histplot(off_diag, color="red", label="Incorrect Pairs (Distractors)", kde=True, stat="density", alpha=0.5)
    
    plt.title(f"Similarity Distribution (MRR: {results['mrr']:.4f})")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "similarity_distribution.png")
    print(f"Saved similarity distribution to {output_path / 'similarity_distribution.png'}")
    
    # 2. Hits@K Plot
    plt.figure(figsize=(8, 5))
    k_vals = ["Hits@1", "Hits@5", "Hits@10"]
    scores = [results["hits@1"], results["hits@5"], results["hits@10"]]
    
    bars = plt.bar(k_vals, scores, color=["blue", "orange", "green"])
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Code Retrieval Accuracy (Hits@K)")
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2%}", ha='center', va='bottom')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path / "hits_at_k.png")
    print(f"Saved Hits@K plot to {output_path / 'hits_at_k.png'}")
    
    # 3. Rank Distribution
    plt.figure(figsize=(10, 6))
    rankings = np.array(results["rankings"])
    sns.histplot(rankings, bins=30, color="purple", kde=False)
    plt.title("Distribution of Correct Code Ranks")
    plt.xlabel("Rank (Lower is better)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / "rank_distribution.png")
    print(f"Saved rank distribution to {output_path / 'rank_distribution.png'}")

def main():
    parser = argparse.ArgumentParser(description="Plot Repo-JEPA evaluation results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    
    args = parser.parse_args()
    plot_evaluation(args.results, args.output_dir)

if __name__ == "__main__":
    main()
