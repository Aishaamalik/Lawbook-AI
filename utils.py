import logging
import os
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOGGING_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration."""
    return logger

def get_file_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file for idempotence."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def list_txt_files(directory: str) -> List[str]:
    """List all .txt files in the directory recursively."""
    return [str(f) for f in Path(directory).rglob("*.txt")]

def normalize_text(text: str) -> str:
    """Normalize text: fix encoding, remove excessive whitespace."""
    # Replace weird control characters
    text = text.replace('\r\n', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\x0c', '')  # form feeds
    text = re.sub(r'\x00+', '', text)  # null bytes
    return text.strip()

def extract_case_numbers(text: str) -> List[str]:
    """Extract case numbers from text using regex."""
    # Patterns like "Constitution Petitions Nos. 11 -15, 18 -22"
    pattern = r'(?:Constitution Petitions?|W\.P\.|C\.A\.)\s*(?:Nos?\.?|No\.?)\s*([\d\s\-,\.]+)'
    matches = re.findall(pattern, text, re.I)
    return [match.strip() for match in matches if match.strip()]

def extract_title(text: str) -> str:
    """Extract case title from text."""
    # Look for lines with VERSUS or v..
    lines = text.split('\n')
    for line in lines[:20]:  # Check first 20 lines
        if 'VERSUS' in line.upper() or ' v. ' in line:
            return line.strip()
    return ""

def extract_judges(text: str) -> List[str]:
    """Extract judges from text."""
    # Look for PRESENT block
    m = re.search(r'PRESENT\s*:\s*(.*?)\n\n', text, re.S | re.I)
    if m:
        block = m.group(1)
        judges = re.findall(r'MR\. JUSTICE ([A-Z\s\-]+)', block, re.I)
        return [judge.strip() for judge in judges]
    return []

def extract_dates(text: str) -> str:
    """Extract hearing dates from text."""
    # Pattern: Dates of hearing: 24-31/5, ... 2010.
    m = re.search(r'Dates of hearing:\s*(.*?)\.', text, re.I)
    if m:
        return m.group(1).strip()
    return ""

def detect_sections(text: str) -> List[Dict[str, str]]:
    """Detect sections in the text."""
    # Common section headers
    section_patterns = [
        r'ORDER',
        r'JUDGMENT',
        r'OPINION',
        r'ARGUMENTS',
        r'FACTS',
        r'ISSUES',
        r'REASONING',
        r'CONCLUSION',
        r'HELD'
    ]
    sections = []
    lines = text.split('\n')
    current_section = "INTRODUCTION"
    current_text = ""
    for line in lines:
        line_upper = line.upper().strip()
        if any(re.match(pattern, line_upper) for pattern in section_patterns):
            if current_text:
                sections.append({"section": current_section, "text": current_text.strip()})
            current_section = line_upper
            current_text = ""
        else:
            current_text += line + '\n'
    if current_text:
        sections.append({"section": current_section, "text": current_text.strip()})
    return sections

def strip_repeated_headers_footers(text: str) -> str:
    """Remove repeated headers and footers."""
    lines = text.split('\n')
    # Identify repeated lines (simple heuristic: lines that appear more than 5 times)
    from collections import Counter
    line_counts = Counter(lines)
    repeated = {line for line, count in line_counts.items() if count > 5 and len(line.strip()) > 10}
    # Remove repeated lines
    cleaned_lines = [line for line in lines if line not in repeated]
    return '\n'.join(cleaned_lines)

def sentence_tokenize(text: str) -> List[str]:
    """Tokenize text into sentences using spaCy or nltk."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [sent.text for sent in doc.sents]
    except ImportError:
        import nltk
        nltk.download('punkt')
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

def chunk_sentences(sentences: List[str], target_tokens: int = 800, overlap_tokens: int = 200) -> List[str]:
    """Chunk sentences into text chunks based on token count."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Simple tokenizer
    chunks = []
    current_chunk = ""
    current_tokens = 0
    overlap_sentences = []

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent))
        if current_tokens + sent_tokens > target_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Prepare overlap
                overlap_sentences = current_chunk.split('.')[-overlap_tokens//50:]  # Rough estimate
                current_chunk = '.'.join(overlap_sentences) + ' ' + sent
                current_tokens = len(tokenizer.encode(current_chunk))
            else:
                current_chunk = sent
                current_tokens = sent_tokens
        else:
            current_chunk += ' ' + sent
            current_tokens += sent_tokens

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_manifest_entry(file_path: str) -> Dict[str, Any]:
    """Build manifest entry for a file."""
    stat = os.stat(file_path)
    return {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": stat.st_size,
        "last_modified": stat.st_mtime,
        "sha256": get_file_sha256(file_path)
    }
