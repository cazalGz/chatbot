from src.rag_pipeline import RAGPipeline
from src.pdf_loader import load_pdfs
from src.embedder import Embedder
from src.vector_db import VectorDB

class ConsoleChat:
    def __init__(self, max_history=5, top_k=3):
        print("[INFO] Inicializando chatbot...")
        # Load PDFs and embeddings
        pdfs = load_pdfs()
        self.embedder = Embedder()
        self.embeddings, self.metadata = self.embedder.embed_pdfs(pdfs)

        # Load o create FAISS index
        self.vector_db = VectorDB()
        try:
            self.vector_db.load_index()
        except FileNotFoundError:
            print("[INFO] Creando índice FAISS...")
            self.vector_db.create_index(self.embeddings.astype("float32"))
            self.vector_db.save_index()

        # create pipeline RAG
        self.rag = RAGPipeline(top_k=top_k)
        self.history = []
        self.max_history = max_history

    def chat(self):
        print("\n[INFO] Chatbot listo. Escribe 'exit' para salir.\n")
        while True:
            query = input("Tú: ").strip()
            if query.lower() == "exit":
                print("Saliendo del chatbot...")
                break
            response = self.rag.answer(query, self.metadata)
            print(f"Bot: {response}\n")
            
            # Save to historial
            self.history.append({"query": query, "response": response})
            if len(self.history) > self.max_history:
                self.history.pop(0)  # keep only last interactions

# quick test
if __name__ == "__main__":
    chat = ConsoleChat()
    chat.chat()
