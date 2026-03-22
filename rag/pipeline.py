import os
from typing import List

from llama_cpp import Llama
from rag.loader import load_documents, prepare_chunks
from rag.embedder import Embedder
from rag.vector_store import VectorStore
import os
from huggingface_hub import hf_hub_download
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
        model_path = os.getenv("MODEL_PATH")

        if not model_path:
            raise ValueError("MODEL_PATH not set in .env")

        model_path = ensure_model(model_path)

        self.llm = Llama(model_path=model_path)

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

def ensure_model(model_path: str):
    """
    Ensure GGUF model exists locally, otherwise download it.
    """
    if os.path.exists(model_path):
        print(f"Model found at: {model_path}")
        return model_path

    print("Model not found. Downloading from Hugging Face...")

    # Create directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Download model
    downloaded_path = hf_hub_download(
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_0.gguf",
        local_dir=os.path.dirname(model_path),
        local_dir_use_symlinks=False
    )

    print(f"Model downloaded to: {downloaded_path}")
    return downloaded_path