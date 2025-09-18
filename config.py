import os

# Global configuration for Legal Document Q&A System

# Dataset folder path
DATASET_FOLDER = os.path.join(os.getcwd(), "Supreme_court_Of_Pakistan_judgments")

# Target chunk length and overlap in tokens
TARGET_CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

# Embedding model name (SentenceTransformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Embedding batch size
EMBEDDING_BATCH_SIZE = 32

# ChromaDB collection name and persist directory
CHROMA_COLLECTION_NAME = "scp_judgments"
CHROMA_PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")

# Logging config
LOGGING_LEVEL = "INFO"

# Ollama / LLaMA2 local inference config
OLLAMA_API_URL = "http://localhost:11434"  # Example local API endpoint for Ollama

# Other configs
MAX_QUERY_RESULTS = 5
