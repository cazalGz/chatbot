import os
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.getenv("DATA_DIR")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR")
MODEL_CACHE_DIR = os.getenv("HF_HOME")  

# -------------------------
# Models
# -------------------------
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

# -------------------------
# LLM configuration
# -------------------------
LLM_LOAD_IN_4BIT = os.getenv("LLM_LOAD_IN_4BIT", "True").lower() == "true"
LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda")

# -------------------------
# FAISS configuration
# -------------------------
VECTOR_DIMENSION = 768
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")

# -------------------------
# Chat parameters
# -------------------------
MAX_HISTORY = int(os.getenv("MAX_HISTORY"))
