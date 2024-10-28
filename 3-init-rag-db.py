#!/usr/bin/env python3

import os
# Set OpenBLAS environment variables to suppress warnings and control threading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from database import Database
from data_ingestion import DataIngestionPipeline
from vectorization import VectorizationPipeline
from loguru import logger
import json
from pymongo import ReplaceOne  # Import ReplaceOne to fix the error

search_index = 'data/search-index.json'

def init_database():
    logger.info("Initializing database...")

    # Initialize database connection
    db = Database()

    # Drop existing collection if it exists
    logger.info("Dropping existing collection...")
    db.collection.drop()

    # Create indices
    logger.info("Creating indices...")
    db.collection.create_index("title")
    db.collection.create_index("content")

    logger.info("Database initialized successfully!")

def load_documents():
    logger.info("Loading and processing documents...")

    # Initialize pipelines
    data_pipeline = DataIngestionPipeline()
    vectorization_pipeline = VectorizationPipeline()

    # Check if documents.json exists
    if not os.path.exists(search_index):
        logger.error(f"{search_index} not found!")
        return

    # Load and process documents
    documents = data_pipeline.load_data(search_index)
    processed_docs = data_pipeline.preprocess_data(documents)

    # Generate embeddings and store documents
    embeddings = vectorization_pipeline.generate_embeddings([doc["content"] for doc in processed_docs])

    # Store documents with embeddings in the database
    batch_store_documents(processed_docs, embeddings)

    # Print summary
    logger.info(f"Processed {len(processed_docs)} documents")

def batch_store_documents(documents, embeddings):
    logger.info("Storing documents in the database...")

    db = Database()  # Ensure a database instance is available
    operations = []

    for doc, embedding in zip(documents, embeddings):
        # Append the ReplaceOne operation with correct structure
        operations.append({
            'replaceOne': {
                'filter': {"url": doc["url"]},
                'replacement': {
                    "url": doc["url"],
                    "title": doc["title"],
                    "content": doc["content"],
                    "embedding": embedding
                },
                'upsert': True  # Create the document if it doesn't exist
            }
        })

    try:
        # Perform bulk_write with properly structured operations using ReplaceOne
        result = db.collection.bulk_write([ReplaceOne(**op['replaceOne']) for op in operations])
        logger.info(f"Bulk write result: {result.bulk_api_result}")
    except Exception as e:
        logger.error(f"Error batch storing documents: {e}")

def main():
    print("\nRAG System Database Initialization")
    print("=================================")

    while True:
        print("\nOptions:")
        print("1. Initialize database (will delete existing data)")
        print("2. Load documents from data/search-index.json")
        print("3. Exit")

        choice = input("\nEnter your choice (1-3): ")

        if choice == '1':
            init_database()
        elif choice == '2':
            load_documents()
        elif choice == '3':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

