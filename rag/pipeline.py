import os
from typing import List

from llama_cpp import Llama
from rag.loader import load_documents, prepare_chunks
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from dotenv import load_dotenv
load_dotenv()
class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm = Llama(model_path=os.getenv("MODEL_PATH"))

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
        Generate answer using local GGUF model (llama.cpp).
        """
        context = "\n\n".join(context_chunks)

        prompt = f"""[INST]

        You are a helpful assistant. Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {query}
        [/INST]
        """
        output = self.llm(
            prompt,
            max_tokens=300,
            temperature=0.3,
            stop=["</s>"]
        )

        return output["choices"][0]["text"].strip()

    def query(self, user_query: str) -> str:
        """
        Full pipeline: retrieve + generate.
        """
        context_chunks = self.retrieve(user_query)
        answer = self.generate_answer(user_query, context_chunks)
        return answer
