import os
import json
from typing import List, Dict, Any
import config
import utils

logger = utils.setup_logging()

class DataIngestion:
    def __init__(self, input_dir: str = config.DATASET_FOLDER):
        self.input_dir = input_dir
        self.manifest_file = os.path.join(os.getcwd(), "processed_manifest.json")
        self.processed_manifest = self.load_manifest()

    def load_manifest(self) -> Dict[str, Any]:
        """Load processed manifest for idempotence."""
        if os.path.exists(self.manifest_file):
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {}

    def save_manifest(self):
        """Save processed manifest."""
        with open(self.manifest_file, 'w') as f:
            json.dump(self.processed_manifest, f, indent=4)

    def discover_files(self) -> List[str]:
        """Discover all .txt files."""
        return utils.list_txt_files(self.input_dir)

    def should_process_file(self, file_path: str) -> bool:
        """Check if file needs processing based on SHA256."""
        sha256 = utils.get_file_sha256(file_path)
        return sha256 not in self.processed_manifest

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a single file: load, normalize, extract basic metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                raw_text = f.read()

        normalized_text = utils.normalize_text(raw_text)

        # Basic metadata
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "sha256": utils.get_file_sha256(file_path),
            "case_numbers": utils.extract_case_numbers(normalized_text),
            "case_name": utils.extract_title(normalized_text),
            "judges": utils.extract_judges(normalized_text),
            "hearing_dates": utils.extract_dates(normalized_text),
            "raw_text": normalized_text
        }

        # Mark as processed
        self.processed_manifest[metadata["sha256"]] = {
            "file_path": file_path,
            "processed_at": str(os.path.getmtime(file_path))
        }

        return metadata

    def run_ingestion(self) -> List[Dict[str, Any]]:
        """Run full ingestion pipeline."""
        files = self.discover_files()
        ingested_data = []

        for file_path in files:
            if self.should_process_file(file_path):
                logger.info(f"Processing {file_path}")
                data = self.process_file(file_path)
                ingested_data.append(data)
            else:
                logger.info(f"Skipping {file_path} (already processed)")

        self.save_manifest()
        return ingested_data

if __name__ == "__main__":
    ingestion = DataIngestion()
    data = ingestion.run_ingestion()
    logger.info(f"Ingested {len(data)} files")
