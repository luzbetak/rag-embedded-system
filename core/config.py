# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the RAG (Retrieval-Augmented Generation) system.
    
    This class centralizes all configuration parameters for the document processing
    and retrieval pipeline, including database connections, model settings, and
    processing parameters.
    
    Attributes:
        MONGODB_URI (str): MongoDB connection string. Defaults to local instance
            if not specified in environment variables.
        DATABASE_NAME (str): Name of the MongoDB database for storing documents
            and their embeddings.
        COLLECTION_NAME (str): Name of the MongoDB collection where documents
            will be stored.
        MODEL_NAME (str): Name/path of the sentence transformer model used for
            generating embeddings. Uses the all-MiniLM-L6-v2 model which provides
            a good balance between performance and accuracy.
        BATCH_SIZE (int): Number of documents to process simultaneously during
            embedding generation. Adjust based on available memory.
        VECTOR_DIMENSION (int): Dimension of the embedding vectors produced by
            the model. Must match the output dimension of the specified MODEL_NAME.
        TOP_K (int): Number of most similar documents to retrieve during the
            search process.
    """
    
    # Database Configuration
    MONGODB_URI     = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME   = "rag_database"
    COLLECTION_NAME = "documents"
    BATCH_SIZE      = 4 
    
    # Model Configuration
    # Models and their dimensions
    # "all-MiniLM-L6-v2"                      -> 384 dimensions
    # "all-mpnet-base-v2"                     -> 768 dimensions
    # "all-MiniLM-L12-v2"                     -> 384 dimensions
    # "paraphrase-multilingual-MiniLM-L12-v2" -> 384 dimensions

    # VECTOR_DIMENSION = 384 is fixed property of the "all-MiniLM-L6-v2" model
    # Each text input will be converted into a vector with exactly 384 numbers
    # It means ANY text - whether it's a single word, a sentence, a paragraph, 
    # or even a chunk of text - will be converted into exactly 384 numbers by the model. 
    # This is called the text embedding.

    MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIMENSION = 384
    
    # TOP_K Retrieval Configuration

    # For very focused answers
    TOP_K = 2  # Only get the two most relevant chunks

    # For more comprehensive answers
    # TOP_K = 5  # Get the five most relevant chunks

    # For broad research
    # TOP_K = 10  # Get many potentially relevant chunks

