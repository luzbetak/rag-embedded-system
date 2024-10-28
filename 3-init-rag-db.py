#!/usr/bin/env python3

import os
import sys
from pathlib import Path
# Set OpenBLAS environment variables to suppress warnings and control threading
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from database import Database
from data_ingestion import DataIngestionPipeline
from vectorization import VectorizationPipeline
from loguru import logger
import json
from pymongo import ReplaceOne

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/rag_indexing.log", rotation="500 MB")

search_index = 'data/search-index.json'

def verify_search_index():
    """Verify the search index file exists and has content"""
    if not os.path.exists(search_index):
        logger.error(f"Search index file not found: {search_index}")
        return False
        
    try:
        with open(search_index, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not data:
                logger.error("Search index file is empty")
                return False
            logger.info(f"Search index contains {len(data)} documents")
            return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in search index: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading search index: {e}")
        return False

def init_database():
    logger.info("Initializing database...")

    try:
        # Initialize database connection
        db = Database()

        # Drop existing collection if it exists
        logger.info("Dropping existing collection...")
        db.collection.drop()

        # Create indices
        logger.info("Creating indices...")
        db.collection.create_index([("url", 1)], unique=True)
        db.collection.create_index([("title", 1)])
        db.collection.create_index([("content", 1)])

        logger.info("Database initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def batch_store_documents(documents, embeddings):
    """Store documents with their embeddings in MongoDB"""
    logger.info("Storing documents in the database...")

    if not documents or not embeddings:
        logger.error("No documents or embeddings to store")
        return

    if len(documents) != len(embeddings):
        logger.error(f"Mismatch between documents ({len(documents)}) and embeddings ({len(embeddings)})")
        return

    db = Database()
    operations = []
    stored_count = 0

    for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
        try:
            # Validate document fields
            if not all(key in doc for key in ["url", "title", "content"]):
                logger.warning(f"Document {idx} missing required fields, skipping")
                continue

            # The embedding is already a list, no need to convert
            document = {
                "url": doc["url"],
                "title": doc["title"],
                "content": doc["content"],
                "embedding": embedding
            }

            operations.append(
                ReplaceOne(
                    {"url": doc["url"]},
                    document,
                    upsert=True
                )
            )
            stored_count += 1

        except Exception as e:
            logger.error(f"Error processing document {idx}: {e}")
            continue

    if not operations:
        logger.error("No valid documents to store")
        return

    try:
        # Perform bulk write operation
        result = db.collection.bulk_write(operations)
        logger.info(f"Successfully stored {stored_count} documents")
        logger.info(f"Bulk write results:")
        logger.info(f"  Inserted: {result.upserted_count}")
        logger.info(f"  Modified: {result.modified_count}")
        logger.info(f"  Matched: {result.matched_count}")

        # Verify final document count
        total_docs = db.collection.count_documents({})
        logger.info(f"Total documents in collection after storage: {total_docs}")

    except Exception as e:
        logger.error(f"Error during bulk write: {e}")

def load_documents():
    """Load and process documents from search index"""
    logger.info("Starting document loading process...")

    # Verify search index first
    if not verify_search_index():
        return

    try:
        # Initialize pipelines
        data_pipeline = DataIngestionPipeline()
        vectorization_pipeline = VectorizationPipeline()

        # Load documents
        documents = data_pipeline.load_data(search_index)
        logger.info(f"Loaded {len(documents)} documents from search index")

        # Process documents
        processed_docs = data_pipeline.preprocess_data(documents)
        logger.info(f"Processed {len(processed_docs)} documents")

        if not processed_docs:
            logger.error("No documents to process")
            return

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = vectorization_pipeline.generate_embeddings(
            [doc["content"] for doc in processed_docs]
        )
        logger.info(f"Generated embeddings for {len(embeddings)} documents")

        # Store documents with embeddings
        batch_store_documents(processed_docs, embeddings)

    except Exception as e:
        logger.error(f"Error in document loading process: {e}")
        raise

def main():
    print("\nRAG System Database Initialization")
    print("=================================")

    while True:
        print("\nOptions:")
        print("1. Initialize database (will delete existing data)")
        print("2. Load documents from data/search-index.json")
        print("3. Show document count")
        print("4. Exit")

        try:
            choice = input("\nEnter your choice (1-4): ")

            if choice == '1':
                init_database()
            elif choice == '2':
                load_documents()
            elif choice == '3':
                db = Database()
                count = db.collection.count_documents({})
                print(f"\nTotal documents in database: {count}")
                if count > 0:
                    sample = db.collection.find_one({}, {'_id': 0, 'embedding': 0})
                    print("\nSample document (excluding embedding):")
                    print(json.dumps(sample, indent=2))
            elif choice == '4':
                print("\nExiting...")
                break
            else:
                print("\nInvalid choice. Please try again.")

            input("\nPress Enter to continue...")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()
