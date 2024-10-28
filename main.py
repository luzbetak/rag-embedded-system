#!/usr/bin/env python

from config import Config
from data_ingestion import DataIngestionPipeline
from vectorization import VectorizationPipeline
from query import QueryEngine
from loguru import logger
import asyncio  # Add this import

def setup_logging():
    logger.add("logs/pipeline.log", rotation="500 MB")

async def main():  # Make this async
    setup_logging()
    logger.info("Starting RAG pipeline")

    try:
        # Initialize pipelines
        data_pipeline = DataIngestionPipeline()
        vectorization_pipeline = VectorizationPipeline()
        query_engine = QueryEngine()

        # Load and process documents
        documents = data_pipeline.load_data("data/documents.json")
        processed_docs = data_pipeline.preprocess_data(documents)

        # Generate embeddings and store
        vectorization_pipeline.process_documents(processed_docs)

        logger.info("Pipeline completed successfully")

        # Test query
        test_query = "Your test query here"
        results = await query_engine.search(test_query)  # Add await here
        logger.info(f"Test query results: {results}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
