# Legal Document Q&A System - Implementation Plan

## Information Gathered
- **Requirements**: Build a Semantic RAG system for 2000+ Supreme Court of Pakistan judgment .txt files. Includes data ingestion, preprocessing, embedding, indexing, query processing, answer generation with Ollama/LLaMA2, Streamlit UI, and FastAPI backend.
- **Dataset**: Local folder `Supreme_court_Of_Pakistan_judgments/` with 2000+ .txt files containing case rulings (facts, issues, reasoning, verdict).
- **Pipeline**: Detailed preprocessing steps including cleaning, metadata extraction, section detection, semantic chunking, deduplication, embedding with SentenceTransformers, and storage in ChromaDB with FAISS.
- **Tech Stack**: Python 3.10+, SentenceTransformers, ChromaDB, FAISS, Ollama, Streamlit, FastAPI, Pandas, NumPy.

## Plan: Detailed Code Update Plan at File Level
1. **Setup Project Structure and Dependencies**
   - Create `requirements.txt` with all necessary libraries.
   - Create `config.py` for global configurations (input folder, chunk length, embedding model, etc.).
   - Create `utils.py` for helper functions (logging, file handling).

2. **Data Ingestion and Preprocessing**
   - Create `data_ingestion.py`: File discovery, checksum for idempotence, raw text loading with encoding fixes.
   - Create `preprocessing.py`: Header/footer removal, metadata extraction (case_id, judges, sections), section detection, sentence tokenization, semantic chunking with overlap.

3. **Embedding and Indexing**
   - Create `embedding.py`: Load SentenceTransformers model, generate embeddings in batches.
   - Create `indexing.py`: Upsert chunks to ChromaDB with metadata, handle deduplication, persist DB.

4. **Query Processing and RAG**
   - Create `query_processor.py`: Accept queries, convert to embeddings, perform similarity search in ChromaDB.
   - Create `rag.py`: Feed retrieved passages to Ollama/LLaMA2 for answer generation, provide citations.

5. **User Interface (Streamlit)**
   - Create `app.py`: Dataset loader, query box, answer panel with citations, highlighted sections, upload option.

6. **Backend (FastAPI)**
   - Create `main.py`: Endpoints for /upload, /ask, /search, /train.

7. **Integration and Testing**
   - Integrate Ollama for local inference.
   - Create `run_pipeline.py` to execute the full data processing pipeline.
   - Test with sample queries, evaluate retrieval relevance and citation accuracy.

## Dependent Files to be Edited/Created
- `requirements.txt` (new)
- `config.py` (new)
- `utils.py` (new)
- `data_ingestion.py` (new)
- `preprocessing.py` (new)
- `embedding.py` (new)
- `indexing.py` (new)
- `query_processor.py` (new)
- `rag.py` (new)
- `app.py` (new, Streamlit)
- `main.py` (new, FastAPI)
- `run_pipeline.py` (new)
- `TODO.md` (this file)

## Followup Steps
- Install dependencies using `pip install -r requirements.txt`.
- Run data processing pipeline with `python run_pipeline.py`.
- Test Streamlit UI with `streamlit run app.py`.
- Test FastAPI backend with `uvicorn main:app --reload`.
- Evaluate system with sample queries and manual checks.
- Optimize for performance (GPU for embeddings, batching).
- Handle incremental updates and re-indexing.
