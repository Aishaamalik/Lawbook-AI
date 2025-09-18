from typing import List, Dict, Any
import config
import utils
from embedding import Embedder
from indexing import Indexer

logger = utils.setup_logging()

class QueryProcessor:
    def __init__(self):
        self.embedder = Embedder()
        self.indexer = Indexer()

    def process_query(self, query: str, top_k: int = config.MAX_QUERY_RESULTS) -> List[Dict[str, Any]]:
        """Process a query: embed, search, return results."""
        # Embed query
        query_embedding = self.embedder.encode_chunks([query])[0]

        # Search in ChromaDB
        results = self.indexer.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            }
            formatted_results.append(result)

        logger.info(f"Retrieved {len(formatted_results)} results for query: {query}")
        return formatted_results

if __name__ == "__main__":
    processor = QueryProcessor()
    results = processor.process_query("What is the law on termination?")
    for res in results:
        print(res["document"])
