"""
Offline wiring test for graph.py / main.py.

This sandbox cannot reach huggingface.co to download all-MiniLM-L6-v2 (network
egress here is restricted to package registries only), so this test stubs the
embedding step with a deterministic fake encoder and pre-populates a local
ChromaDB collection, then exercises the REAL classify_intent /
retrieve_and_answer / direct_answer / route_by_intent / FastAPI code paths
end-to-end. Run ingest.py + a real request against main.py on a machine with
open internet access (which downloads the real MiniLM model on first run) for
the actual graded artifact.
"""
import os
import sys
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MOCK_LLM"] = "1"

import chromadb
import graph as graph_module
import ingest as ingest_module

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store_test")


class FakeEncoder:
    """Deterministic fake embedding: hash text -> fixed-size vector, so the
    same text always maps to the same vector (good enough for wiring tests;
    NOT semantically meaningful)."""
    def encode(self, texts):
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(h, dtype=np.uint8).astype(float)
            v = np.tile(v, 3)[:384]  # match MiniLM's 384 dims
            vecs.append((v / np.linalg.norm(v)).tolist())
        return np.array(vecs)


def setup_fake_index():
    chunks = ingest_module.load_and_chunk_documents()

    encoder = FakeEncoder()
    embeddings = encoder.encode([c["text"] for c in chunks]).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection("zepto_policies")
    except Exception:
        pass
    coll = client.create_collection("zepto_policies")
    coll.add(ids=[c["chunk_id"] for c in chunks],
             documents=[c["text"] for c in chunks],
             embeddings=embeddings)
    return encoder


def main():
    encoder = setup_fake_index()

    # Monkeypatch graph.py's model/collection getters to use the fake encoder
    # and the test ChromaDB dir, WITHOUT changing graph.py's real logic.
    graph_module._embed_model = encoder
    graph_module._get_embed_model = lambda: encoder

    def fake_get_collection():
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_collection("zepto_policies")
    graph_module._get_collection = fake_get_collection
    graph_module._collection = None

    print("=== Example call 1: policy question (should trigger retrieval) ===")
    resp1 = graph_module.run_query("What is your refund policy?")
    print(resp1.model_dump_json(indent=2))
    assert resp1.sources, "policy_question should have populated sources"

    print("\n=== Example call 2: general question (should NOT trigger retrieval) ===")
    resp2 = graph_module.run_query("What's the weather like today?")
    print(resp2.model_dump_json(indent=2))
    assert resp2.sources == [], "general_question should have empty sources"
    assert "only answer questions about Zepto" in resp2.answer

    print("\nAll wiring assertions passed: classify_intent routing, conditional "
          "edges, retrieval, mock-mode canned answers, and schema validation "
          "all work end-to-end.")


if __name__ == "__main__":
    main()
