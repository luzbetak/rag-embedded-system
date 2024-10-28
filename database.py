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
            document = {
                "title": doc.get('title', ''),
                "content": doc.get('content', ''),
                "metadata": doc.get('metadata', {}),
                "embedding": embedding
            }
            result = self.collection.insert_one(document)
            logger.info(f"Stored document with id: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise

    def batch_store_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]] = None):
        """Store multiple documents with their embeddings"""
        try:
            docs_to_insert = []
            for idx, doc in enumerate(documents):
                embedding = embeddings[idx] if embeddings else None
                document = {
                    "title": doc.get('title', ''),
                    "content": doc.get('content', ''),
                    "metadata": doc.get('metadata', {}),
                    "embedding": embedding
                }
                docs_to_insert.append(document)
            
            if docs_to_insert:
                # Remove any existing documents with the same titles
                existing_titles = [doc['title'] for doc in docs_to_insert]
                self.collection.delete_many({"title": {"$in": existing_titles}})
                # Insert new documents
                result = self.collection.insert_many(docs_to_insert)
                logger.info(f"Stored {len(result.inserted_ids)} documents")
        except Exception as e:
            logger.error(f"Error batch storing documents: {str(e)}")
            raise

    def get_similar_documents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents using cosine similarity"""
        try:
            # Get all documents
            documents = list(self.collection.find({}, {'_id': 0}))
            results = []
            
            # Calculate similarities if we have embeddings
            if query_embedding:
                query_embedding = np.array(query_embedding)
                for doc in documents:
                    doc_embedding = doc.pop('embedding', None)  # Remove embedding from result
                    if doc_embedding:
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
            else:
                # If no embeddings, return documents without sorting
                for doc in documents:
                    doc.pop('embedding', None)  # Remove embedding from result
                    doc['score'] = 0.0
                    results.append(doc)
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
