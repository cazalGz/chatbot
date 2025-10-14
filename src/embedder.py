from sentence_transformers import SentenceTransformer
from src.config import EMBEDDINGS_MODEL
import numpy as np

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits the text into fragments of 'chunk_size' words, with 'overlap' overlapping words.
    Returns a list of strings.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

class Embedder:
    def __init__(self, model_name=EMBEDDINGS_MODEL):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Receives a list of strings (text fragments) and returns embeddings as a numpy array.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings

    def embed_pdfs(self, pdf_dict):
        """
        Receives a dictionary {pdf_name: full_text} and returns:
        - list of embeddings
        - list of metadata for each fragment: {"source": pdf_name, "chunk_index": i, "text": fragment}
        """
        all_chunks = []
        metadata = []

        for pdf_name, text in pdf_dict.items():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            for i, chunk in enumerate(chunks):
                metadata.append({"source": pdf_name, "chunk_index": i, "text": chunk})

        print(f"[INFO] Total fragments to generate embeddings: {len(all_chunks)}")
        embeddings = self.embed_texts(all_chunks)
        return embeddings, metadata

# Quick test
if __name__ == "__main__":
    from src.pdf_loader import load_pdfs
    pdfs = load_pdfs()
    embedder = Embedder()
    embeddings, metadata = embedder.embed_pdfs(pdfs)
    print(f"[INFO] Embeddings generated: {embeddings.shape}")
