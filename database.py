#!/usr/bin/env python3

from typing import List, Dict, Any
from pymongo import MongoClient
import numpy as np
from config import Config
from loguru import logger

class Database:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        logger.info("Initialized MongoDB database connection")

    def store_document(self, doc: Dict[str, Any], embedding: List[float] = None):
        """Store a single document with its embedding"""
        try:
            # Validate required fields
            if not doc.get('url'):
                logger.error("Skipping document: Missing URL")
                return
                
            document = {
                "url": doc['url'].strip(),  # Primary key
                "title": doc.get('title', '').strip(),
                "content": doc.get('content', '').strip(),
                "embedding": embedding
            }
            
            # Use upsert to handle duplicates
            result = self.collection.update_one(
                {"url": document["url"]},  # Query by URL
                {"$set": document},        # Update/insert document
                upsert=True                # Create if doesn't exist
            )
            
            logger.info(f"{'Updated' if result.matched_count else 'Inserted'} document with URL: {document['url']}")
            return result

        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    def batch_store_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]] = None):
        """Store multiple documents with their embeddings"""
        try:
            docs_to_insert = []
            for idx, doc in enumerate(documents):
                # Skip documents without URL
                if not doc.get('url'):
                    logger.warning(f"Skipping document at index {idx}: Missing URL")
                    continue
                    
                embedding = embeddings[idx] if embeddings else None
                
                document = {
                    "url": doc['url'].strip(),
                    "title": doc.get('title', '').strip(),
                    "content": doc.get('content', '').strip(),
                    "embedding": embedding
                }
                docs_to_insert.append(document)

            if not docs_to_insert:
                logger.warning("No valid documents to insert")
                return

            # Use bulk write with upsert operations
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
            
            result = self.collection.bulk_write(operations)
            logger.info(f"Processed {len(docs_to_insert)} documents. "
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
