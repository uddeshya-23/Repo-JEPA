"""
Linear Probe Evaluator

Validates JEPA embeddings by training a simple classifier on frozen features.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

from ..model import RepoJEPA
from ..data import CodeSearchNetDataset, CodeDocCollator, get_tokenizer


class LinearProbeEvaluator:
    """
    Linear probe evaluator for validating JEPA embeddings.
    
    Extracts frozen embeddings and trains a simple linear classifier
    to test if embeddings contain semantic information.
    """
    
    def __init__(
        self,
        model: RepoJEPA,
        tokenizer,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def extract_embeddings(
        self,
        dataset: CodeSearchNetDataset,
        batch_size: int = 32,
        max_samples: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings from frozen encoder.
        
        Returns:
            embeddings: [N, D] numpy array
            labels: [N] numpy array (based on simple heuristics)
        """
        collator = CodeDocCollator(self.tokenizer)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        
        all_embeddings = []
        all_labels = []
        
        for batch in tqdm(loader, desc="Extracting embeddings"):
            code_input_ids = batch["code_input_ids"].to(self.device)
            code_attention_mask = batch["code_attention_mask"].to(self.device)
            
            # Get code embeddings
            embeddings = self.model.encode_code(code_input_ids, code_attention_mask)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if max_samples and sum(len(e) for e in all_embeddings) >= max_samples:
                break
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        if max_samples:
            embeddings = embeddings[:max_samples]
        
        # Generate pseudo-labels based on code patterns
        # This is a simple heuristic for demonstration
        labels = self._generate_pseudo_labels(dataset, len(embeddings))
        
        return embeddings, labels
    
    def _generate_pseudo_labels(
        self,
        dataset: CodeSearchNetDataset,
        n_samples: int,
    ) -> np.ndarray:
        """
        Generate pseudo-labels based on code patterns.
        
        Categories:
            0: File I/O (open, read, write)
            1: Network (request, http, url)
            2: Data Processing (sort, filter, map)
            3: Math (sum, mean, calculate)
            4: Other
        """
        patterns = {
            0: ["open", "read", "write", "file", "path"],
            1: ["request", "http", "url", "fetch", "api"],
            2: ["sort", "filter", "map", "list", "array"],
            3: ["sum", "mean", "calculate", "math", "number"],
        }
        
        labels = []
        for i in range(min(n_samples, len(dataset))):
            item = dataset[i]
            code = item["code"].lower()
            
            label = 4  # Default: Other
            for cat, keywords in patterns.items():
                if any(kw in code for kw in keywords):
                    label = cat
                    break
            
            labels.append(label)
        
        return np.array(labels)
    
    def train_probe(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train and evaluate linear probe.
        
        Returns:
            Dict with accuracy and per-class metrics
        """
        # Train logistic regression
        clf = LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )
        
        clf.fit(train_embeddings, train_labels)
        
        # Predict
        train_preds = clf.predict(train_embeddings)
        test_preds = clf.predict(test_embeddings)
        
        # Metrics
        train_acc = accuracy_score(train_labels, train_preds)
        test_acc = accuracy_score(test_labels, test_preds)
        
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "n_train": len(train_labels),
            "n_test": len(test_labels),
        }
    
    def evaluate(
        self,
        train_dataset: CodeSearchNetDataset,
        test_dataset: CodeSearchNetDataset,
        max_train_samples: int = 5000,
        max_test_samples: int = 1000,
    ) -> Dict[str, float]:
        """
        Full linear probe evaluation pipeline.
        """
        print("Extracting training embeddings...")
        train_embeds, train_labels = self.extract_embeddings(
            train_dataset, max_samples=max_train_samples
        )
        
        print("Extracting test embeddings...")
        test_embeds, test_labels = self.extract_embeddings(
            test_dataset, max_samples=max_test_samples
        )
        
        print("Training linear probe...")
        results = self.train_probe(
            train_embeds, train_labels,
            test_embeds, test_labels,
        )
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Linear Probe Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--max-train", type=int, default=5000, help="Max training samples")
    parser.add_argument("--max-test", type=int, default=1000, help="Max test samples")
    
    args = parser.parse_args()
    
    # Load model
    from ..config import RepoJEPAConfig
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = RepoJEPAConfig(**checkpoint.get("config", {}))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepoJEPA(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    tokenizer = get_tokenizer(config.pretrained_encoder)
    
    # Load datasets
    train_dataset = CodeSearchNetDataset(split="train", max_samples=args.max_train)
    test_dataset = CodeSearchNetDataset(split="test", max_samples=args.max_test)
    
    # Evaluate
    evaluator = LinearProbeEvaluator(model, tokenizer, device)
    results = evaluator.evaluate(train_dataset, test_dataset)
    
    print("\n" + "=" * 50)
    print("📊 LINEAR PROBE RESULTS")
    print("=" * 50)
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy:  {results['test_accuracy']:.4f}")
    print(f"N Train: {results['n_train']}")
    print(f"N Test:  {results['n_test']}")
    print("=" * 50)
    
    if results['test_accuracy'] >= 0.85:
        print("✅ SUCCESS: Test accuracy >= 85% (target achieved)")
    else:
        print(f"⚠️  Accuracy below target (85%). Current: {results['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
