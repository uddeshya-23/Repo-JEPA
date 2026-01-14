
from src.utils.search import RepoJEPASearch
import os

def demo():
    # 1. Initialize
    searcher = RepoJEPASearch("uddeshya-k/RepoJepa")
    
    # 2. Load the index we just created
    if not os.path.exists("project.index"):
        print("Error: project.index not found. Did the indexer fail?")
        return
        
    searcher.load_index("project.index")
    
    # 3. Define some interesting semantic queries
    queries = [
        "how is the vicreg loss calculated?",
        "where do we update the target encoder using EMA?",
        "how are docstrings used to train the model?"
    ]
    
    for q in queries:
        print(f"\n" + "="*60)
        print(f"🔍 QUERY: \"{q}\"")
        print("="*60)
        
        results = searcher.query(q, top_k=1)
        if results:
            res = results[0]
            print(f"✅ Best Match (Score: {res['score']:.4f})")
            print(f"📍 Location: {res['file']}:{res['line']}")
            print("-" * 20)
            # Print first 10 lines of the code snippet for brevity
            code_lines = res['code'].splitlines()
            print("\n".join(code_lines[:15]))
            if len(code_lines) > 15:
                print("... [truncated]")
            print("-" * 20)

if __name__ == "__main__":
    demo()
