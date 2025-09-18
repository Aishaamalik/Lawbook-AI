import requests
from typing import List, Dict, Any
import config
import utils

logger = utils.setup_logging()

class RAG:
    def __init__(self, ollama_url: str = config.OLLAMA_API_URL):
        self.ollama_url = ollama_url

    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> dict:
        """Generate answer using RAG with Ollama."""
        # Prepare context from retrieved documents
        context = "\n\n".join([doc["document"] for doc in retrieved_docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context:"

        # Call Ollama API
        payload = {
            "model": "llama2",  # Assuming LLaMA2 is loaded
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "No answer generated.")
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            answer = "Error in generating answer."

        # Add citations
        citations = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            case_name = meta.get('case_name', 'Unknown Case')
            section = meta.get('section', 'Unknown Section')
            citation = f"{case_name}, {section}"
            citations.append(citation)

        return {"answer": answer, "citations": citations}

if __name__ == "__main__":
    rag = RAG()
    sample_query = "What is the law on termination?"
    sample_docs = [{"document": "Sample doc", "metadata": {"case_name": "Test Case", "section": "Reasoning"}}]
    answer = rag.generate_answer(sample_query, sample_docs)
    print(answer)
