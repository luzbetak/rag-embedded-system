#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import json
from loguru import logger
from pymongo import ReplaceOne
from database import Database
from data_ingestion import DataIngestionPipeline
from vectorization import VectorizationPipeline

# Control threading and suppress warnings
os.environ.update({
    'OPENBLAS_NUM_THREADS': '1',
    'OPENBLAS_MAIN_FREE': '1',
    'OMP_NUM_THREADS': '1'
})

# Configure logging
logger.remove()
logger.add(sys.stderr, 
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/rag_indexing.log", rotation="500 MB")

class RAGDatabaseInitializer:
    def __init__(self, search_index_path='data/search-index.json'):
        self.search_index = search_index_path
        self.db = Database()

    def verify_search_index(self):
        """Verify search index file exists and contains valid data"""
        if not os.path.exists(self.search_index):
            logger.error(f"Search index not found: {self.search_index}")
            return False
            
        try:
            with open(self.search_index, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    logger.error("Empty search index")
                    return False
                logger.info(f"Found {len(data)} documents in search index")
                return True
        except Exception as e:
            logger.error(f"Error reading search index: {e}")
            return False

    def init_database(self):
        """Initialize database with required indices"""
        logger.info("Initializing database...")
        try:
            self.db.collection.drop()
            self.db.collection.create_index([("url", 1)], unique=True)
            self.db.collection.create_index([("title", 1)])
            self.db.collection.create_index([("content", 1)])
            logger.info("Database initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            return False

    def store_documents(self, documents, embeddings):
        """Store documents with embeddings in MongoDB"""
        if not self._validate_inputs(documents, embeddings):
            return

        operations = self._prepare_operations(documents, embeddings)
        if not operations:
            return

        self._execute_bulk_write(operations)

    def _validate_inputs(self, documents, embeddings):
        """Validate input documents and embeddings"""
        if not documents or not embeddings:
            logger.error("Missing documents or embeddings")
            return False
        if len(documents) != len(embeddings):
            logger.error(f"Document/embedding count mismatch: {len(documents)} vs {len(embeddings)}")
            return False
        return True

    def _prepare_operations(self, documents, embeddings):
        """Prepare MongoDB operations for bulk write"""
        operations = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            try:
                if not all(key in doc for key in ["url", "title", "content"]):
                    logger.warning(f"Document {idx} missing required fields")
                    continue
                
                operations.append(
                    ReplaceOne(
                        {"url": doc["url"]},
                        {
                            "url": doc["url"],
                            "title": doc["title"],
                            "content": doc["content"],
                            "embedding": embedding
                        },
                        upsert=True
                    )
                )
            except Exception as e:
                logger.error(f"Error processing document {idx}: {e}")
                continue
        return operations

    def _execute_bulk_write(self, operations):
        """Execute bulk write operations and log results"""
        try:
            result = self.db.collection.bulk_write(operations)
            logger.info(f"Bulk write completed:")
            logger.info(f"  Inserted: {result.upserted_count}")
            logger.info(f"  Modified: {result.modified_count}")
            logger.info(f"  Matched: {result.matched_count}")
            
            total_docs = self.db.collection.count_documents({})
            logger.info(f"Total documents in collection: {total_docs}")
        except Exception as e:
            logger.error(f"Bulk write error: {e}")

    def load_documents(self):
        """Load and process documents from search index"""
        if not self.verify_search_index():
            return

        try:
            data_pipeline = DataIngestionPipeline()
            vectorization_pipeline = VectorizationPipeline()

            documents = data_pipeline.load_data(self.search_index)
            processed_docs = data_pipeline.preprocess_data(documents)
            
            if not processed_docs:
                logger.error("No documents to process")
                return

            embeddings = vectorization_pipeline.generate_embeddings(
                [doc["content"] for doc in processed_docs]
            )
            
            self.store_documents(processed_docs, embeddings)

        except Exception as e:
            logger.error(f"Document loading error: {e}")
            raise

def main():
    rag_init = RAGDatabaseInitializer()
    
    menu_options = {
        "1": ("Initialize database (will delete existing data)", rag_init.init_database),
        "2": ("Load documents from data/search-index.json", rag_init.load_documents),
        "3": ("Show document count", lambda: print_document_count(rag_init.db)),
        "4": ("Exit", sys.exit)
    }

    def print_document_count(db):
        count = db.collection.count_documents({})
        print(f"\nTotal documents in database: {count}")
        if count > 0:
            sample = db.collection.find_one({}, {'_id': 0, 'embedding': 0})
            print("\nSample document (excluding embedding):")
            print(json.dumps(sample, indent=2))

    print("\nRAG System Database Initialization")
    print("=================================")

    while True:
        try:
            print("\nOptions:")
            for key, (description, _) in menu_options.items():
                print(f"{key}. {description}")

            choice = input("\nEnter your choice (1-4): ")
            
            if choice in menu_options:
                menu_options[choice][1]()
                if choice != "4":
                    input("\nPress Enter to continue...")
            else:
                print("\nInvalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")

if __name__ == "__main__":
    main()
