from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class Embedder:
    """
    Handles text embeddings using a local sentence-transformers model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of texts into embeddings.
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        """
        return self.model.encode(query, convert_to_numpy=True)
