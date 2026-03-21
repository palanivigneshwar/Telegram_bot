import os
from typing import List

from openai import OpenAI

from rag.loader import load_documents, prepare_chunks
from rag.embedder import Embedder
from rag.vector_store import VectorStore

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self._initialize()

    def _initialize(self):
        """
        Load documents, create chunks, generate embeddings, and store them.
        """
        print("Loading documents...")
        documents = load_documents(self.data_path)

        print("Chunking documents...")
        chunks = prepare_chunks(documents)

        print(f"Total chunks: {len(chunks)}")

        print("Generating embeddings...")
        embeddings = self.embedder.embed_texts(chunks)

        print("Storing embeddings...")
        self.vector_store.add(embeddings, chunks)

        print("RAG pipeline ready!")

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k relevant chunks.
        """
        query_vec = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vec, top_k=top_k)

        return [text for text, _ in results]

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generate answer using OpenAI with retrieved context.
        """
        context = "\n\n".join(context_chunks)

        prompt = f"""
        You are a helpful assistant. Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {query}

        Answer clearly and concisely.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def query(self, user_query: str) -> str:
        """
        Full pipeline: retrieve + generate.
        """
        context_chunks = self.retrieve(user_query)
        answer = self.generate_answer(user_query, context_chunks)
        return answer
