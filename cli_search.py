
import os
import sys
from src.utils.search import RepoJEPASearch

def main():
    print("\n" + "="*60)
    print("🚀 REPO-JEPA INTERACTIVE SEARCH CLI")
    print("="*60)
    
    # 1. Initialize
    try:
        searcher = RepoJEPASearch("uddeshya-k/RepoJepa")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load Index
    index_file = "project.index"
    if not os.path.exists(index_file):
        print(f"❌ '{index_file}' not found. Please run the indexer first:")
        print("python -m src.utils.indexer --path . --save project.index")
        return
        
    try:
        searcher.load_index(index_file)
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return

    print("\n✅ Ready! Type your query below (or 'exit' to quit).")
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye! 👋")
                break
                
            print(f"Searching for: '{query}'...")
            results = searcher.query(query, top_k=3)
            
            if not results:
                print("No matches found.")
                continue
                
            for i, res in enumerate(results):
                print(f"\n[{i+1}] 🏆 Match Score: {res['score']:.4f}")
                print(f"📍 Location: {res['file']}:{res['line']}")
                print("-" * 30)
                # Show source code snippet
                code_lines = res['code'].splitlines()
                # Show first 8 lines
                print("\n".join(code_lines[:8]))
                if len(code_lines) > 8:
                    print(f"... [{len(code_lines)-8} more lines]")
                print("-" * 30)
                
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Search error: {e}")

if __name__ == "__main__":
    main()
