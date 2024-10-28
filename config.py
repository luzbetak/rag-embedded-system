# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME = "rag_database"
    COLLECTION_NAME = "documents"
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 32
    VECTOR_DIMENSION = 384
    TOP_K = 5
