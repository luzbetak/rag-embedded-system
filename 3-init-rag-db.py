#!/usr/bin/env python3

from typing import List, Dict, Any
from pymongo import MongoClient, UpdateOne
import numpy as np
from config import Config
from loguru import logger

class Database:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        logger.info("Initialized MongoDB database connection")

    def batch_store_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]] = None):
        """Store multiple documents with their embeddings"""
        try:
            # Prepare bulk operations
            operations = []
            for idx, doc in enumerate(documents):
                # Skip documents without URL
                if not doc.get('url'):
                    logger.warning(f"Skipping document at index {idx}: Missing URL")
                    continue
                    
                document = {
                    "url": doc['url'].strip(),
                    "title": doc.get('title', '').strip(),
                    "content": doc.get('content', '').strip(),
                    "embedding": embeddings[idx] if embeddings else None
                }
                
                # Create update operation
                operation = UpdateOne(
                    {"url": document["url"]},  # Filter by URL
                    {"$set": document},        # Update/insert document
                    upsert=True                # Create if doesn't exist
                )
                operations.append(operation)

            if not operations:
                logger.warning("No valid documents to insert")
                return

            # Execute bulk write
            result = self.collection.bulk_write(operations)
            logger.info(f"Successfully processed {len(operations)} documents. "
                       f"Inserted: {result.upserted_count}, "
                       f"Modified: {result.modified_count}")

        except Exception as e:
            logger.error(f"Error batch storing documents: {str(e)}")
            raise

    def get_similar_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents using cosine similarity"""
        try:
            # Get all documents with embeddings
            documents = list(self.collection.find(
                {"embedding": {"$exists": True}},
                {'_id': 0}
            ))
            
            results = []
            query_embedding = np.array(query_embedding)
            
            for doc in documents:
                doc_embedding = doc.pop('embedding', None)
                if doc_embedding is not None:
                    similarity = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                    )
                    doc['score'] = float(similarity)
                else:
                    doc['score'] = 0.0
                results.append(doc)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            result = self.collection.delete_many({})
            logger.info(f"Cleared {result.deleted_count} documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
