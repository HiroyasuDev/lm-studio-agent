"""
RAG Knowledge Pipeline for LM Studio Agent
───────────────────────────────────────────
Ingest documents → embed → store → semantic search.

Uses:
  - ChromaDB (file-based vector store, no server needed)
  - LM Studio's nomic-embed-text-v1.5 (already downloaded, 84 MB)
  - Supports: PDF, DOCX, TXT, Markdown

Usage:
  py rag_pipeline.py ingest /path/to/docs     # ingest a folder
  py rag_pipeline.py ingest /path/to/file.pdf  # ingest a single file
  py rag_pipeline.py search "your query"        # semantic search
  py rag_pipeline.py status                     # show collection stats
  py rag_pipeline.py clear                      # wipe the database
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Optional

try:
    import chromadb
except ImportError:
    print("ERROR: chromadb not installed. Run: py -m pip install chromadb")
    sys.exit(1)

# ── Configuration ────────────────────────────────────────────
CHROMA_DIR = Path(r"D:\Local\Tools\LM_Studio\knowledge_base")
COLLECTION_NAME = "local_knowledge"
EMBEDDING_URL = "http://<IP>:<PORT>/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
CHUNK_TOKENS = 512      # Precise cl100k_base tokens per chunk
MAX_RESULTS = 5         # default search results

LOG_DIR = Path(r"D:\Local\Tools\LM_Studio\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"rag_{datetime.now():%Y%m%d}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("rag")


# ═══════════════════════════════════════════════════════════════
#  Document Loaders
# ═══════════════════════════════════════════════════════════════

def load_text_file(path: Path) -> str:
    """Load a plain text or markdown file."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    """Load a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except ImportError:
        log.warning("pypdf not installed. Skipping PDF: %s", path.name)
        return ""
    except Exception as e:
        log.error("Failed to load PDF %s: %s", path.name, e)
        return ""


def load_docx(path: Path) -> str:
    """Load a DOCX file."""
    try:
        from docx import Document
        doc = Document(str(path))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text
    except ImportError:
        log.warning("python-docx not installed. Skipping DOCX: %s", path.name)
        return ""
    except Exception as e:
        log.error("Failed to load DOCX %s: %s", path.name, e)
        return ""


def load_document(path: Path) -> Optional[str]:
    """Load a document based on extension."""
    ext = path.suffix.lower()
    loaders = {
        ".txt": load_text_file,
        ".md": load_text_file,
        ".py": load_text_file,
        ".json": load_text_file,
        ".csv": load_text_file,
        ".sas": load_text_file,
        ".pdf": load_pdf,
        ".docx": load_docx,
    }
    loader = loaders.get(ext)
    if not loader:
        log.debug("Unsupported file type: %s", ext)
        return None
    return loader(path)


# ═══════════════════════════════════════════════════════════════
#  Text Chunking
# ═══════════════════════════════════════════════════════════════

try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKENIZER = None

def chunk_text(text: str, filename: str = "", chunk_tokens: int = CHUNK_TOKENS) -> List[str]:
    """Code-aware and semantic token-based chunking."""
    if not text or not text.strip():
        return []
    
    ext = Path(filename).suffix.lower() if filename else ""
    chunks = []
    
    # 1. Code-Aware Chunking (Python AST precision)
    if ext == ".py":
        try:
            import ast
            tree = ast.parse(text)
            current_chunk = []
            for node in tree.body:
                node_text = ast.get_source_segment(text, node)
                if node_text:
                    current_chunk.append(node_text)
            if current_chunk:
                return current_chunk
        except Exception:
            pass # Syntax error in file, fallback to semantic token chunking
            
    # 2. Code-Aware Chunking (SAS procedural boundaries)
    if ext == ".sas":
        import re
        parts = re.split(r'(?i)\n(?=proc |data |%macro )', text)
        return [p.strip() for p in parts if len(p.strip()) > 20]
        
    # 3. True Tokenizer Semantic Chunking (For Markdown, PDF, TXT)
    if TOKENIZER:
        tokens = TOKENIZER.encode(text)
        start = 0
        while start < len(tokens):
            end = min(start + chunk_tokens, len(tokens))
            chunks.append(TOKENIZER.decode(tokens[start:end]))
            start += chunk_tokens - 50 # 50 token overlap
        return chunks
        
    # 4. Fallback crude character chunking if tiktoken fails
    char_chunk = int(chunk_tokens * 3.5)
    start = 0
    while start < len(text):
        end = start + char_chunk
        chunks.append(text[start:end].strip())
        start = end - 150
    return chunks


# ═══════════════════════════════════════════════════════════════
#  Embedding via LM Studio API
# ═══════════════════════════════════════════════════════════════

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings from LM Studio's embedding endpoint."""
    try:
        payload = json.dumps({
            "model": EMBEDDING_MODEL,
            "input": texts,
        }).encode()

        req = urllib.request.Request(
            EMBEDDING_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())

        embeddings = [item["embedding"] for item in resp["data"]]
        return embeddings

    except urllib.error.URLError:
        log.error("Cannot reach embedding endpoint. Is LM Studio running with an embedding model?")
        log.error("Load it with: lms load nomic-embed-text-v1.5 -y")
        raise
    except Exception as e:
        log.error("Embedding failed: %s", e)
        raise


# ═══════════════════════════════════════════════════════════════
#  ChromaDB Vector Store
# ═══════════════════════════════════════════════════════════════

def get_collection():
    """Get or create the ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_file(path: Path, collection) -> int:
    """Ingest a single file into the vector store. Returns chunk count."""
    text = load_document(path)
    if not text:
        return 0

    chunks = chunk_text(text, filename=path.name)
    if not chunks:
        log.debug("No chunks from: %s", path.name)
        return 0

    # Generate IDs based on file path + chunk index
    ids = [
        hashlib.md5(f"{path}:{i}".encode()).hexdigest()
        for i in range(len(chunks))
    ]

    # Get embeddings in batches of 32
    all_embeddings = []
    for batch_start in range(0, len(chunks), 32):
        batch = chunks[batch_start:batch_start + 32]
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)

    # Upsert into ChromaDB
    metadatas = [
        {
            "source": str(path),
            "filename": path.name,
            "chunk_index": i,
            "ingested_at": datetime.now().isoformat(),
        }
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    log.info("  ✓ %s → %d chunks", path.name, len(chunks))
    return len(chunks)


def ingest_path(target: str) -> None:
    """Ingest a file or directory into the knowledge base."""
    path = Path(target)
    collection = get_collection()

    if path.is_file():
        count = ingest_file(path, collection)
        log.info("Ingested %d chunks from %s", count, path.name)

    elif path.is_dir():
        total = 0
        files = list(path.rglob("*"))
        supported = [f for f in files if f.is_file() and f.suffix.lower() in
                     {".txt", ".md", ".py", ".json", ".csv", ".sas", ".pdf", ".docx"}]
        log.info("Found %d supported files in %s", len(supported), path)

        for f in supported:
            try:
                count = ingest_file(f, collection)
                total += count
            except Exception as e:
                log.error("Failed to ingest %s: %s", f.name, e)

        log.info("Total: %d chunks from %d files", total, len(supported))
    else:
        log.error("Path not found: %s", target)


# ═══════════════════════════════════════════════════════════════
#  Semantic Search
# ═══════════════════════════════════════════════════════════════

def search(query: str, n_results: int = MAX_RESULTS) -> List[dict]:
    """Search the knowledge base. Returns list of {text, source, score}."""
    collection = get_collection()

    if collection.count() == 0:
        log.warning("Knowledge base is empty. Ingest documents first.")
        return []

    query_embedding = get_embeddings([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
    )

    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("filename", "unknown"),
            "score": round(1 - results["distances"][0][i], 3),  # cosine similarity
        })

    return hits


def print_search_results(query: str, n_results: int = MAX_RESULTS):
    """Search and pretty-print results."""
    print(f"\n  Query: {query}")
    print(f"  {'─' * 56}")

    hits = search(query, n_results)
    if not hits:
        print("  No results. Ingest documents first.")
        return

    for i, hit in enumerate(hits, 1):
        print(f"\n  [{i}] Score: {hit['score']:.3f} | Source: {hit['source']}")
        preview = hit["text"][:200].replace("\n", " ")
        print(f"      {preview}...")


def show_status():
    """Show knowledge base statistics."""
    collection = get_collection()
    count = collection.count()
    size_mb = sum(
        f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file()
    ) / (1024 * 1024) if CHROMA_DIR.exists() else 0

    print(f"\n  Knowledge Base Status")
    print(f"  {'─' * 40}")
    print(f"  Chunks:    {count}")
    print(f"  DB size:   {size_mb:.1f} MB")
    print(f"  Location:  {CHROMA_DIR}")
    print(f"  Model:     {EMBEDDING_MODEL}")


def clear_db():
    """Clear the entire knowledge base."""
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        log.info("Knowledge base cleared.")
    else:
        log.info("Knowledge base already empty.")


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG Knowledge Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("path", help="File or directory to ingest")

    p_search = sub.add_parser("search", help="Search knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("-n", type=int, default=MAX_RESULTS, help="Number of results")

    sub.add_parser("status", help="Show stats")
    sub.add_parser("clear", help="Clear database")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_path(args.path)
    elif args.command == "search":
        print_search_results(args.query, args.n)
    elif args.command == "status":
        show_status()
    elif args.command == "clear":
        clear_db()


if __name__ == "__main__":
    main()
