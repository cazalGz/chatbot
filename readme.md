# RAG Chatbot for Course Notes

A local chatbot that analyzes lecture notes and PDFs for a specific course to generate summaries and answer specific questions using the console as an interface.

## 📝 Description

This project implements a chatbot based on the **RAG** ​​(Retrieval-Augmented Generation) architecture. Its function is to process a collection of documents (lecture notes, PDFs, etc.) to build a knowledge base and answer questions or generate summaries about them.

The conversation is stored in RAM to maintain context throughout the session. The system is optimized for local GPU execution using `bitsandbytes` for quantization and `accelerate` for model loading. The orchestration of the RAG logic is handled with the `LangChain` framework.

- **Language Model (LLM):** `mistralai/Mistral-7B-Instruct-v0.2`
- **Embeddings Model:** `sentence-transformers/all-mpnet-base-v2`
- **Vector Database:** `FAISS`

-----

## 🚀 Workflow

1. **Data Loading:** The bot reads all `.pdf` files located in the `/data` directory.
2. **Processing:** It divides the documents into chunks, generates embeddings for each, and indexes them in a FAISS vector database created at runtime.
3. **Query:** The user asks a question through the console.
4. **Retrieval:** The system converts the question into an embedding and searches the FAISS database for the most relevant chunks.
5. **Generation:** The retrieved fragments are sent as context to the LLM along with the original question to generate a coherent response.

-----

## 💻 System Requirements

The chatbot has been tested and works correctly on a computer with the following specifications. This is considered a reference configuration for adequate performance.

- **CPU:** Intel Core i5-10400
- **GPU:** NVIDIA GTX 1060 6GB VRAM
- **RAM:** 16 GB
- **Software:** It is **essential** to have **CUDA Toolkit 11.x** installed for proper operation with the GPU.

-----

## ⚙️ Installation and Running

1. Clone this repository to your local machine.
2. It is highly recommended to create and activate a Python virtual environment to isolate dependencies.
3. Install all necessary dependencies by running:
```bash
pip install -r requirements.txt
```
4. Add your PDF files to the `data/` folder (create it if it doesn't exist).
5. Run the chatbot with the following command:
```bash
python main.py
```

-----

## 📦 Key Dependencies

Although the project has multiple dependencies, the most important are:

- `torch`, `transformers`, `accelerate`: For executing the language model in PyTorch and optimizing it.
- `bitsandbytes`: For 8-bit quantization of the model, reducing VRAM consumption.
- `langchain`: Orchestrates the entire RAG flow, from document loading to response generation.
- `sentence-transformers`: Provides the model for generating text embeddings.
- `faiss-cpu`: Manages the vector database for similarity searching.
- `PyPDF2`: For extracting text from PDF documents.