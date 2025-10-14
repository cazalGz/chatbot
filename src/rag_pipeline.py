from src.vector_db import VectorDB
from src.llm_handler import LLMHandler
from src.embedder import Embedder
import numpy as np

class RAGPipeline:
    def __init__(self, top_k=3):
        self.vector_db = VectorDB()
        self.llm = LLMHandler()
        self.embedder = Embedder()
        self.top_k = top_k

        # Try to load the existing FAISS index
        try:
            self.vector_db.load_index()
            print("[INFO] RAG Pipeline ready with loaded index")
        except FileNotFoundError:
            print("[WARN] FAISS index not found. You must create it first using embedder + vector_db")

    def get_context(self, query):
        """
        Given a query text, returns the most relevant fragments from the FAISS index.
        """
        query_embedding = self.embedder.model.encode([query], convert_to_numpy=True)
        dists, idxs = self.vector_db.search(query_embedding.astype(np.float32), top_k=self.top_k)
        return idxs  # return the indices of the most relevant fragments

    def generate_prompt(self, query, fragments, metadata):
        """
        Builds the prompt that will be sent to the LLM, including context.
        """
        context_texts = [metadata[i]["text"] for i in fragments]
        context_str = "\n\n".join(context_texts)
        prompt = (
            f"Use the following context to answer the question clearly and concisely:\n\n"
            f"{context_str}\n\nQuestion: {query}\nAnswer:"
        )
        return prompt

    def answer(self, query, metadata):
        """
        Generates an answer to the query using RAG.
        """
        fragment_idxs = self.get_context(query)
        prompt = self.generate_prompt(query, fragment_idxs, metadata)
        response = self.llm.generate(prompt)
        return response

# Quick test
if __name__ == "__main__":
    from src.pdf_loader import load_pdfs
    from src.embedder import Embedder

    # Load PDFs and embeddings to populate FAISS if it doesn't exist
    pdfs = load_pdfs()
    embedder = Embedder()
    embeddings, metadata = embedder.embed_pdfs(pdfs)

    from src.vector_db import VectorDB
    db = VectorDB()
    db.create_index(embeddings.astype(np.float32))
    db.save_index()

    # Create RAG pipeline
    rag = RAGPipeline(top_k=3)
    question = "What is Python and what is it used for?"
    response = rag.answer(question, metadata)
    print("\n--- RAG Response ---\n")
    print(response)
