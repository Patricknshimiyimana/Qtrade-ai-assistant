import chromadb
from chromadb.utils import embedding_functions
from src.config import EMBEDDING_MODEL, TOP_K


class InMemoryVectorStore:
    """An ephemeral, zero-persistence vector store running strictly in RAM."""

    def __init__(self):
        # EphemeralClient guarantees no disk junk is generated per our scoping decision
        self.client = chromadb.EphemeralClient()

        # Runs locally via sentence-transformers
        self.embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        )

        # Force cosine distance so the configured escalation distance threshold behaves predictably
        self.collection = self.client.get_or_create_collection(
            name="qtrade_knowledge_base",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_docs(
        self, ids: list[str], documents: list[str], metadatas: list[dict]
    ) -> None:
        """Hydrates the in-memory collection."""
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, query_text: str, top_k: int = TOP_K) -> list[dict]:
        """Queries the vector space and unpacks Chroma's ugly nested lists into clean dicts."""
        raw_results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not raw_results["documents"] or not raw_results["documents"][0]:
            return []

        clean_results = []
        for doc, meta, dist in zip(
            raw_results["documents"][0],
            raw_results["metadatas"][0],
            raw_results["distances"][0],
        ):
            clean_results.append(
                {
                    "excerpt": doc,
                    "source_doc": meta["source_doc"],  # e.g. "Doc 1: Returns & Refunds"
                    "distance": float(dist),  # Cosine distance: 0.0 is exact match
                }
            )

        return clean_results