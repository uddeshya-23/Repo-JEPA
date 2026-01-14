
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

class RepoJEPASearch:
    """
    High-level search engine for Repo-JEPA.
    
    Examples:
        >>> searcher = RepoJEPASearch("uddeshya-23/repo-jepa")
        >>> searcher.add_code(["def hello(): print('world')"])
        >>> searcher.query("greet the world")
    """
    
    def __init__(self, model_path: str, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Repo-JEPA from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device).eval()
        
        self.code_database = []
        self.code_embeddings = None
        
    def add_code(self, snippets: List[str]):
        """Encode and add code snippets to the searchable database."""
        print(f"Encoding {len(snippets)} snippets...")
        new_embeddings = []
        
        for i in range(0, len(snippets), 32):
            batch = snippets[i : i+32]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                embeds = self.model.encode_code(
                    inputs.input_ids, inputs.attention_mask
                )
                new_embeddings.append(embeds.cpu())
                
        self.code_database.extend(snippets)
        if self.code_embeddings is None:
            self.code_embeddings = torch.cat(new_embeddings)
        else:
            self.code_embeddings = torch.cat([self.code_embeddings, torch.cat(new_embeddings)])
            
    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for the most relevant code snippets for a natural language query."""
        inputs = self.tokenizer(
            text, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            query_embed = self.model.encode_query(
                inputs.input_ids, inputs.attention_mask
            )
            
        # Compute cosine similarity
        query_embed = F.normalize(query_embed.cpu(), dim=-1)
        db_embeds = F.normalize(self.code_embeddings, dim=-1)
        
        similarities = (query_embed @ db_embeds.T).squeeze(0)
        scores, indices = torch.topk(similarities, min(top_k, len(self.code_database)))
        
        results = []
        for score, idx in zip(scores, indices):
            results.append((self.code_database[idx.item()], score.item()))
            
        return results

if __name__ == "__main__":
    # Example usage (local test)
    # searcher = RepoJEPASearch("hf_export")
    # searcher.add_code(["def add(a, b): return a + b"])
    # print(searcher.query("sum two numbers"))
    pass
