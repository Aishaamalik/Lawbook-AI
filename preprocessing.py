from typing import List, Dict, Any
import config
import utils

logger = utils.setup_logging()

class Preprocessor:
    def __init__(self):
        pass

    def preprocess_text(self, raw_text: str) -> str:
        """Full preprocessing: clean headers/footers, etc."""
        # Strip repeated headers/footers
        cleaned = utils.strip_repeated_headers_footers(raw_text)
        # Further cleaning if needed
        return cleaned

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract detailed metadata."""
        return {
            "case_numbers": utils.extract_case_numbers(text),
            "case_name": utils.extract_title(text),
            "judges": utils.extract_judges(text),
            "hearing_dates": utils.extract_dates(text)
        }

    def detect_and_chunk_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect sections and chunk them."""
        sections = utils.detect_sections(text)
        chunked_sections = []

        for section in sections:
            sentences = utils.sentence_tokenize(section["text"])
            chunks = utils.chunk_sentences(sentences, config.TARGET_CHUNK_SIZE, config.CHUNK_OVERLAP)
            for i, chunk in enumerate(chunks):
                chunked_sections.append({
                    "section": section["section"],
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "start_char": text.find(chunk),  # Approximate
                    "end_char": text.find(chunk) + len(chunk)
                })

        return chunked_sections

    def process_document(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single document: preprocess, extract, chunk."""
        raw_text = metadata["raw_text"]
        cleaned_text = self.preprocess_text(raw_text)
        detailed_metadata = self.extract_metadata(cleaned_text)
        chunks = self.detect_and_chunk_sections(cleaned_text)

        # Combine metadata with chunks
        processed_chunks = []
        for chunk in chunks:
            chunk_metadata = {
                **metadata,
                **detailed_metadata,
                **chunk
            }
            processed_chunks.append(chunk_metadata)

        return processed_chunks

if __name__ == "__main__":
    # Example usage
    preprocessor = Preprocessor()
    # Assume metadata from ingestion
    sample_metadata = {
        "file_path": "sample.txt",
        "raw_text": "Sample text with ORDER section."
    }
    chunks = preprocessor.process_document(sample_metadata)
    logger.info(f"Processed into {len(chunks)} chunks")
