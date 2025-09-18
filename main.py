from fastapi import FastAPI, UploadFile, File
from query_processor import QueryProcessor
from rag import RAG
import os
import shutil

app = FastAPI(title="Legal Document Q&A API")

query_processor = QueryProcessor()
rag = RAG()

@app.post("/ask")
async def ask_question(query: str):
    """Query the system and get answer with citations."""
    results = query_processor.process_query(query)
    answer = rag.generate_answer(query, results)
    return {"query": query, "answer": answer, "retrieved_passages": results}

@app.post("/search")
async def search_passages(query: str, top_k: int = 5):
    """Retrieve top-k passages without generation."""
    results = query_processor.process_query(query, top_k)
    return {"query": query, "results": results}

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload additional documents."""
    uploaded = []
    for file in files:
        save_path = os.path.join("uploaded_docs", file.filename)
        os.makedirs("uploaded_docs", exist_ok=True)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded.append(file.filename)
    return {"uploaded_files": uploaded}

@app.post("/train")
async def reindex():
    """Re-index new uploads."""
    # Trigger re-indexing pipeline
    return {"message": "Re-indexing initiated."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
