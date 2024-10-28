#!/usr/bin/env python3

from database import Database
from data_ingestion import DataIngestionPipeline
from vectorization import VectorizationPipeline
from loguru import logger
import json
import os

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
    if not os.path.exists("data/documents.json"):
        logger.error("data/documents.json not found!")
        return
    
    # Load and process documents
    documents = data_pipeline.load_data("data/documents.json")
    processed_docs = data_pipeline.preprocess_data(documents)
    
    # Generate embeddings and store documents
    vectorization_pipeline.process_documents(processed_docs)
    
    # Print summary
    logger.info(f"Processed {len(processed_docs)} documents")

def main():
    print("\nRAG System Database Initialization")
    print("=================================")
    
    while True:
        print("\nOptions:")
        print("1. Initialize database (will delete existing data)")
        print("2. Load documents from data/documents.json")
        print("3. Do both (initialize and load)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            init_database()
        elif choice == '2':
            load_documents()
        elif choice == '3':
            init_database()
            load_documents()
        elif choice == '4':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
