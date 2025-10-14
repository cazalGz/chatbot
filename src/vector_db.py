import os
import faiss
import numpy as np
from src.config import FAISS_INDEX_PATH, VECTOR_DIMENSION

class VectorDB:
    def __init__(self, index_path=FAISS_INDEX_PATH, dim=VECTOR_DIMENSION):
        self.index_path = index_path
        self.dim = dim
        self.index = None

    def create_index(self, embeddings):
        """
        Creates a FAISS index from the embeddings (numpy array).
        """
        print("[INFO] Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(embeddings)
        print(f"[INFO] Total indexed vectors: {self.index.ntotal}")

    def save_index(self):
        """
        Saves the FAISS index to disk.
        """
        if self.index is None:
            raise ValueError("No index to save. Create or load one first.")
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"[INFO] FAISS index saved to {self.index_path}")

    def load_index(self):
        """
        Loads an existing FAISS index from disk.
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index not found at {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        print(f"[INFO] FAISS index loaded from {self.index_path}")

    def search(self, query_embedding, top_k=5):
        """
        Searches for the 'top_k' nearest vectors to query_embedding.
        Returns: (distances, indices)
        """
        if self.index is None:
            raise ValueError("No index loaded. Create or load one first.")
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0]

# Quick test
if __name__ == "__main__":
    from src.embedder import Embedder
    from src.pdf_loader import load_pdfs

    # Load PDFs
    pdfs = load_pdfs()
    embedder = Embedder()
    embeddings, metadata = embedder.embed_pdfs(pdfs)

    # Create index
    db = VectorDB()
    db.create_index(embeddings.astype('float32'))  # FAISS requires float32
    db.save_index()

    # Example search
    sample_query = embeddings[0]  # we search for the first embedding as a test
    dists, idxs = db.search(sample_query, top_k=3)
    print(f"[INFO] Distances: {dists}")
    print(f"[INFO] Indices: {idxs}")