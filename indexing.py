import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict, Any
import config
import utils

logger = utils.setup_logging()

class Indexer:
    def __init__(self, collection_name: str = config.CHROMA_COLLECTION_NAME, persist_dir: str = config.CHROMA_PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(f"Initialized ChromaDB collection: {collection_name}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Upsert chunks with embeddings and metadata to ChromaDB."""
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk['sha256']}_{chunk['section']}_{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["chunk_text"])
            metadata = {
                "file_name": chunk["file_name"],
                "file_path": chunk["file_path"],
                "case_name": chunk["case_name"],
                "case_number": str(chunk["case_numbers"]),
                "year": "",  # Extract year if available
                "judges": str(chunk["judges"]),
                "section": chunk["section"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "chunk_index": chunk["chunk_index"],
                "sha256_file": chunk["sha256"],
                "hearing_dates": chunk["hearing_dates"]
            }
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks to ChromaDB")

    def persist(self):
        """Persist the database."""
        self.client.persist()
        logger.info("ChromaDB persisted")

if __name__ == "__main__":
    indexer = Indexer()
    # Example
    sample_chunks = [{"sha256": "abc", "section": "ORDER", "chunk_index": 0, "chunk_text": "Sample", "file_name": "test.txt", "file_path": "path", "case_name": "", "case_numbers": [], "judges": [], "hearing_dates": "", "start_char": 0, "end_char": 10}]
    sample_embeddings = [[0.1] * 768]  # Dummy
    indexer.upsert_chunks(sample_chunks, sample_embeddings)
    indexer.persist()
