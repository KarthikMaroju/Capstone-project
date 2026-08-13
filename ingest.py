"""
Module 3 - Support Assistant: Task 1 - Ingestion (chunk + embed + store).

Loads all 8 corpus documents, chunks them (one chunk per document, which is
plenty given their length), embeds each chunk locally with
sentence-transformers/all-MiniLM-L6-v2 (no API key, no network account),
and stores the embeddings in a persistent ChromaDB collection.

Run: python ingest.py
"""
import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")
CHROMA_DIR = os.path.join(HERE, "chroma_store")
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_LOCAL_MODEL_DIR = os.path.join(HERE, "local_models", EMBED_MODEL_NAME)
# If a local copy of the model is present (e.g. this repo's local_models/,
# used as a fallback in network-restricted environments), use it directly
# and skip the network call entirely. Otherwise resolve by hub name, which
# downloads (and caches) from huggingface.co normally.
EMBED_MODEL = _LOCAL_MODEL_DIR if os.path.isdir(_LOCAL_MODEL_DIR) else EMBED_MODEL_NAME


def load_and_chunk_documents():
    """One chunk per document -- each doc_XX.txt is already a single short,
    self-contained policy paragraph, so further splitting would only hurt
    retrieval coherence."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))):
        doc_id = os.path.splitext(os.path.basename(path))[0]  # e.g. "doc_01"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        chunks.append({"chunk_id": doc_id, "text": text})
    return chunks


def build_index():
    chunks = load_and_chunk_documents()
    print(f"Loaded {len(chunks)} document chunks")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode([c["text"] for c in chunks]).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
    )
    print(f"Indexed {collection.count()} chunks into ChromaDB at {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
