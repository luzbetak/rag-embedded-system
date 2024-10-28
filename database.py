#!/usr/bin/env python3

from pymongo import MongoClient, ReplaceOne
from loguru import logger
import numpy as np

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
                    'embedding': embedding.tolist()  # Convert numpy array to list for MongoDB storage
                }
                docs_to_insert.append(doc_to_insert)

        # Check if there are valid documents to insert
        if not docs_to_insert:
            logger.warning("No valid documents to insert")
            return

        # Prepare the bulk write operations with upsert
        operations = [
            ReplaceOne(
                {'url': doc['url']},
                doc,
                upsert=True
            )
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

    def get_similar_documents(self, query_embedding, top_k=5):
        """
        Find similar documents using vector similarity search
        
        Args:
            query_embedding (np.ndarray): The embedding vector of the query
            top_k (int): Number of similar documents to return
            
        Returns:
            list: List of similar documents with their similarity scores
        """
        try:
            # Convert query embedding to list for MongoDB comparison
            query_embedding = query_embedding.tolist()

            # Aggregate pipeline for vector similarity search
            pipeline = [
                {
                    "$addFields": {
                        "similarity": {
                            "$reduce": {
                                "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                "initialValue": 0,
                                "in": {
                                    "$add": [
                                        "$$value",
                                        {"$multiply": ["$$this.0", "$$this.1"]}
                                    ]
                                }
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

            # Execute the aggregation pipeline
            results = list(self.collection.aggregate(pipeline))
            
            if not results:
                logger.warning("No similar documents found")
                return []

            logger.info(f"Found {len(results)} similar documents")
            return results

        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return []

    def close(self):
        """Close the database connection"""
        self.client.close()
