from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from embedding import Embedder
from indexing import Indexer
import utils

logger = utils.setup_logging()

def run_full_pipeline():
    """Run the complete data processing pipeline."""
    logger.info("Starting data ingestion...")
    ingestion = DataIngestion()
    ingested_data = ingestion.run_ingestion()

    logger.info("Starting preprocessing...")
    preprocessor = Preprocessor()
    all_chunks = []
    for data in ingested_data:
        chunks = preprocessor.process_document(data)
        all_chunks.extend(chunks)

    logger.info("Starting embedding generation...")
    embedder = Embedder()
    chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    embeddings = embedder.encode_chunks(chunk_texts)

    logger.info("Starting indexing...")
    indexer = Indexer()
    indexer.upsert_chunks(all_chunks, embeddings)
    indexer.persist()

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_full_pipeline()
