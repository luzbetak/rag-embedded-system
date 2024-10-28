#!/usr/bin/env python3

from pymongo import MongoClient, ReplaceOne
from loguru import logger
import numpy as np

class Database:
    def __init__(self):
        """Initialize MongoDB client and connect to the database"""
        try:
            # Connect with explicit database name
            self.client = MongoClient("mongodb://localhost:27017/")
            
            # Use 'rag_database' explicitly
            self.db = self.client.rag_database
            self.collection = self.db.documents
            
            # Test connection
            self.client.server_info()
            
            # Create indices if they don't exist
            self.collection.create_index([("url", 1)], unique=True)
            self.collection.create_index([("title", 1)])
            self.collection.create_index([("content", 1)])
            
            logger.info(f"Connected to MongoDB - Database: {self.db.name}, Collection: {self.collection.name}")
            
            # Log current document count
            count = self.collection.count_documents({})
            logger.info(f"Current document count: {count}")
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def get_similar_documents(self, query_embedding, top_k=5):
        """Find similar documents using vector similarity search"""
        try:
            # Ensure query_embedding is a list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()

            # Modified aggregation pipeline to handle array multiplication properly
            pipeline = [
                {
                    "$set": {
                        "similarity": {
                            "$let": {
                                "vars": {
                                    "dot_product": {
                                        "$reduce": {
                                            "input": {"$range": [0, {"$size": "$embedding"}]},
                                            "initialValue": 0,
                                            "in": {
                                                "$add": [
                                                    "$$value",
                                                    {
                                                        "$multiply": [
                                                            {"$arrayElemAt": ["$embedding", "$$this"]},
                                                            {"$arrayElemAt": [query_embedding, "$$this"]}
                                                        ]
                                                    }
                                                ]
                                            }
                                        }
                                    }
                                },
                                "in": "$$dot_product"
                            }
                        }
                    }
                },
                {"$sort": {"similarity": -1}},
                {"$limit": top_k},
                {
                    "$project": {
                        "_id": 0,
                        "title": 1,
                        "content": 1,
                        "url": 1,
                        "score": "$similarity"
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar documents")
            
            # Log similarity scores for debugging
            if results:
                logger.debug("Top similarity scores:")
                for i, doc in enumerate(results[:3], 1):
                    logger.debug(f"Doc {i}: {doc.get('score', 0):.3f}")
            
            return results

        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def batch_store_documents(self, documents, embeddings):
        """Store multiple documents with their embeddings"""
        if not documents or not embeddings:
            logger.error("No documents or embeddings to store")
            return

        operations = []
        for doc, embedding in zip(documents, embeddings):
            if not all(key in doc for key in ["url", "title", "content"]):
                logger.warning(f"Skipping document missing required fields: {doc}")
                continue

            # Ensure embedding is a list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

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

        if operations:
            try:
                result = self.collection.bulk_write(operations)
                logger.info(f"Bulk write completed successfully")
                logger.info(f"Documents processed: {len(operations)}")
                logger.info(f"Documents inserted: {result.upserted_count}")
                logger.info(f"Documents modified: {result.modified_count}")
                
                # Verify storage
                count = self.collection.count_documents({})
                logger.info(f"Total documents in collection: {count}")
                
                return result
            except Exception as e:
                logger.error(f"Error storing documents: {e}")
                raise

    def close(self):
        """Close the database connection"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("Database connection closed")


