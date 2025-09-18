from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import config
import utils

logger = utils.setup_logging()

class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def encode_chunks(self, chunks: List[str], batch_size: int = config.EMBEDDING_BATCH_SIZE) -> List[np.ndarray]:
        """Encode text chunks into embeddings."""
        embeddings = self.model.encode(chunks, batch_size=batch_size, show_progress_bar=True)
        logger.info(f"Encoded {len(chunks)} chunks")
        return embeddings.tolist()  # Convert to list for JSON serialization

if __name__ == "__main__":
    embedder = Embedder()
    sample_chunks = ["This is a sample chunk.", "Another chunk for testing."]
    embeddings = embedder.encode_chunks(sample_chunks)
    logger.info(f"Embeddings shape: {len(embeddings)} x {len(embeddings[0])}")
