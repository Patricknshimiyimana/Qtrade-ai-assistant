from pathlib import Path
from src.vector_store import InMemoryVectorStore

# map raw text filenames to Appendix A citation titles to guarantee a valid "[Cited: X]" tag.
DOC_SOURCE_MAP = {
    "doc1_returns.txt": "Doc 1: Returns & Refunds",
    "doc2_shipping.txt": "Doc 2: Shipping",
    "doc3_smarthub.txt": "Doc 3: SmartHub Setup & Troubleshooting",
    "doc4_warranty.txt": "Doc 4: Warranty",
}


def build_ephemeral_store() -> InMemoryVectorStore:
    """Instantiates the RAM store, reads the 4 local txt files, and indexes them."""
    store = InMemoryVectorStore()
    raw_docs_dir = Path(__file__).parent.parent / "data" / "raw_docs"

    ids = []
    documents = []
    metadatas = []

    for file_path in raw_docs_dir.glob("*.txt"):
        file_name = file_path.name
        if file_name not in DOC_SOURCE_MAP:
            continue

        content = file_path.read_text(encoding="utf-8").strip()
        official_title = DOC_SOURCE_MAP[file_name]

        # Scoping Judgment: Because the files are 3-4 sentences long,
        # splitting them degrades semantic context. The document is the chunk.
        ids.append(file_name)
        documents.append(content)
        metadatas.append({"source_doc": official_title})

    store.upsert_docs(ids=ids, documents=documents, metadatas=metadatas)
    return store