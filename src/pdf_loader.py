import os
from PyPDF2 import PdfReader
from src.config import DATA_DIR

def load_pdfs(data_dir=DATA_DIR):
    """
    Reads all PDFs in the data_dir folder and returns a dictionary:
    { "file_name": "full content of the PDF" }
    """
    pdf_texts = {}
    
    # Listing all PDF files
    for file_name in os.listdir(data_dir):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(data_dir, file_name)
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                pdf_texts[file_name] = text.strip()
                print(f"[INFO] PDF loaded: {file_name}")
            except Exception as e:
                print(f"[ERROR] Could not read {file_name}: {e}")
    
    if not pdf_texts:
        print("[WARN] No PDFs found in the data/ folder")

    return pdf_texts

# Quick test
if __name__ == "__main__":
    pdfs = load_pdfs()
    for name, content in pdfs.items():
        print(f"\n--- {name} ---\n{content[:300]}...\n")
