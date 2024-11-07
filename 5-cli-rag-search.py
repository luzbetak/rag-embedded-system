#!/usr/bin/env python3
import os
import asyncio
from query import QueryEngine
from loguru import logger

# Set OpenBLAS environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()

    def print_results(self, results):
        print("\nSearch Results:")
        print("-" * 50)
        if not results:
            print("\nNo documents found.")
            return

        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"Title: {doc.get('title', 'N/A')}")
            print(f"Content: {doc.get('content', 'N/A')}")
            if 'metadata' in doc:
                print("Metadata:")
                for key, value in doc['metadata'].items():
                    print(f"  {key}: {value}")
            if 'score' in doc:
                print(f"Similarity Score: {doc['score']:.3f}")
            print("-" * 30)

    async def search_loop(self):
        print("\nRAG CLI Search")
        print("=" * 50)
        print("Enter 'exit' to quit")
        print("=" * 50)

        while True:
            query = input("\nEnter search query: ")

            if query.lower() == 'exit':
                print("\nGoodbye!")
                break

            print("\nSearching...")
            try:
                results = await self.query_engine.search(query)
                self.print_results(results)
            except Exception as e:
                print(f"\nError during search: {str(e)}")

def main():
    searcher = CLISearch()
    asyncio.run(searcher.search_loop())

if __name__ == "__main__":
    main()
