import os
from typing import List

def load_documents(data_path: str) -> List[str]:
    """
    Load all .txt and .md documents from a directory.
    """
    documents = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    for filename in os.listdir(data_path):
        if filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(data_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)

    return documents

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def prepare_chunks(documents: List[str]) -> List[str]:
    """
    Convert list of documents into list of chunks.
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    return all_chunks