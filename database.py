#!/usr/bin/env python3

from pymongo import MongoClient, ReplaceOne
from loguru import logger

class Database:
    def __init__(self):
        # Initialize MongoDB client and connect to the database
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client['rag_database']
        self.collection = self.db['documents']
        logger.info("Initialized MongoDB database connection")

    def batch_store_documents(self, documents, embeddings):
        docs_to_insert = []
        # Combine documents with their corresponding embeddings
        for doc, embedding in zip(documents, embeddings):
            # Ensure the document has required fields (url, title, and content)
            if doc.get('url') and doc.get('title') and doc.get('content'):
                doc_to_insert = {
                    'url': doc['url'],
                    'title': doc['title'],
                    'content': doc['content'],
                    'embedding': embedding
                }
                docs_to_insert.append(doc_to_insert)

        # Check if there are valid documents to insert
        if not docs_to_insert:
            logger.warning("No valid documents to insert")
            return

        # Prepare the bulk write operations with upsert
        operations = [
            {
                'replaceOne': {
                    'filter': {'url': doc['url']},
                    'replacement': doc,
                    'upsert': True
                }
            }
            for doc in docs_to_insert
        ]

        try:
            # Perform the bulk write operation
            result = self.collection.bulk_write(operations)
            logger.info(f"Processed {len(docs_to_insert)} documents. "
                        f"Inserted: {result.upserted_count}, "
                        f"Modified: {result.modified_count}")
        except Exception as e:
            logger.error(f"Error batch storing documents: {e}")


def load_documents():
    from vectorization import VectorizationPipeline
    from data_ingestion import load_data, preprocess_data

    # Initialize database and vectorization pipeline
    db = Database()
    vectorization_pipeline = VectorizationPipeline()

    # Load and preprocess data
    logger.info("Loading and processing documents...")
    processed_docs = preprocess_data(load_data("data/search-index.json"))

    # Generate embeddings for the processed documents
    embeddings = vectorization_pipeline.generate_embeddings([doc["content"] for doc in processed_docs])

    # Store the documents and embeddings in the database
    db.batch_store_documents(processed_docs, embeddings)


def init_database():
    # Initialize the MongoDB database (deletes old data and creates new indices)
    logger.info("Initializing database...")
    db = Database()
    
    # Drop the existing collection if it exists
    logger.info("Dropping existing collection...")
    db.collection.drop()
    
    # Create indices on the 'url' field
    logger.info("Creating indices...")
    db.collection.create_index("url", unique=True)

    logger.info("Database initialized successfully!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG System Database Initialization")
    parser.add_argument('--init', action='store_true', help='Initialize the database (will delete existing data)')
    parser.add_argument('--load', action='store_true', help='Load documents from data/search-index.json')
    args = parser.parse_args()

    if args.init:
        init_database()

    if args.load:
        load_documents()


if __name__ == "__main__":
    main()

