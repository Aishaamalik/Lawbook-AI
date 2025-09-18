import os
import pickle
from data_ingestion import DataIngestion
from preprocessing import Preprocessor
from embedding import Embedder
import utils
import config
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_full_pipeline():
    # Step 1: Ingest data
    ingestion = DataIngestion()
    ingested_data = ingestion.run_ingestion()
    logger.info(f"Ingested {len(ingested_data)} files")

    # Step 2: Preprocess and chunk data
    preprocessor = Preprocessor()
    all_chunks = []
    for metadata in ingested_data:
        chunks = preprocessor.process_document(metadata)
        all_chunks.extend(chunks)
    logger.info(f"Processed into {len(all_chunks)} chunks")

    # Step 3: Embed chunks
    embedder = Embedder()
    chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]
    embeddings = embedder.encode_chunks(chunk_texts)
    logger.info(f"Generated embeddings for {len(embeddings)} chunks")

    # Step 4: Save chunks and embeddings to .pkl file
    output_path = os.path.join(os.getcwd(), "processed_data.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"chunks": all_chunks, "embeddings": embeddings}, f)
    logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    run_full_pipeline()
