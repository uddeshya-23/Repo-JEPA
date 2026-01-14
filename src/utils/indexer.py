
import os
import ast
import torch
from pathlib import Path
from tqdm import tqdm
from .search import RepoJEPASearch

class RepoIndexer:
    """
    Automates the scanning and indexing of a local codebase.
    """
    
    def __init__(self, searcher: RepoJEPASearch):
        self.searcher = searcher

    def _extract_functions(self, file_path):
        """Extracts individual functions from a Python file using AST."""
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                tree = ast.parse(f.read())
            except:
                return []
                
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Get the full source of the function
                start_line = node.lineno
                end_line = getattr(node, "end_lineno", start_line + 5)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    func_code = "".join(lines[start_line-1:end_line])
                    functions.append({
                        "name": node.name,
                        "code": func_code,
                        "file": str(file_path),
                        "line": start_line
                    })
        return functions

    def index_project(self, project_dir: str):
        """Recursively scans a directory and indexes all Python functions."""
        project_path = Path(project_dir)
        py_files = list(project_path.rglob("*.py"))
        
        all_snippets = []
        metadata = []
        
        print(f"🔍 Scanning {len(py_files)} files in {project_dir}...")
        
        for py_file in tqdm(py_files, desc="Parsing files"):
            # Skip virtual environments or hidden folders
            if any(part.startswith('.') or part == 'venv' for part in py_file.parts):
                continue
                
            funcs = self._extract_functions(py_file)
            for f in funcs:
                all_snippets.append(f["code"])
                metadata.append(f)
        
        print(f"🚀 Found {len(all_snippets)} functions. Encoding to JEPA space...")
        self.searcher.add_code(all_snippets)
        self.searcher.metadata = metadata # Attach file/line info
        
    def save(self, path: str):
        """Saves the generated embeddings and metadata for fast future loading."""
        data = {
            "embeddings": self.searcher.code_embeddings,
            "metadata": getattr(self.searcher, "metadata", []),
            "raw_code": self.searcher.code_database
        }
        torch.save(data, path)
        print(f"✅ Index saved to {path}. Ready for near-instant searching next time.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to local project")
    parser.add_argument("--save", type=str, default="repo.index", help="Output index filename")
    args = parser.parse_args()
    
    # Initialize-
    searcher = RepoJEPASearch("uddeshya-k/RepoJepa")
    indexer = RepoIndexer(searcher)
    
    # Run
    indexer.index_project(args.path)
    indexer.save(args.save)
