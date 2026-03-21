import numpy as np
from typing import List, Tuple

class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    """
    def __init__(self):
        self.embeddings = None  # shape: (N, D)
        self.texts = []         # list of chunks

    def add(self, embeddings: np.ndarray, texts: List[str]):
        """
        Store embeddings and corresponding texts.
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have the same length")

        self.embeddings = embeddings
        self.texts = texts

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query vector and all stored vectors.
        """
        # Normalize
        query_norm = query_vec / np.linalg.norm(query_vec)
        matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

        # Dot product
        similarities = np.dot(matrix_norm, query_norm)

        return similarities

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar text chunks.
        Returns list of (text, score)
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []

        similarities = self._cosine_similarity(query_vec, self.embeddings)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.texts[idx], float(similarities[idx])))

        return results
