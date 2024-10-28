#!/usr/bin/env python3
import os
import asyncio
import aiohttp
from loguru import logger

class APISearch:
    def __init__(self):
        self.api_url = "http://localhost:8000/search"
    
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
        print("\nRAG API Search")
        print("=" * 50)
        print("Enter 'exit' to quit")
        print("=" * 50)
        
        async with aiohttp.ClientSession() as session:
            while True:
                query = input("\nEnter search query: ")
                
                if query.lower() == 'exit':
                    print("\nGoodbye!")
                    break
                    
                print("\nSearching...")
                try:
                    async with session.post(
                        self.api_url,
                        json={"text": query, "top_k": 3}
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.print_results(data["similar_documents"])
                            print("\nGenerated Response:")
                            print(data["generated_response"])
                        else:
                            print(f"\nError: API request failed with status {response.status}")
                            error_text = await response.text()
                            print(f"Details: {error_text}")
                except aiohttp.ClientError as e:
                    print(f"\nError connecting to API server: {str(e)}")
                    print("Make sure the API server is running at http://localhost:8000")
                except Exception as e:
                    print(f"\nUnexpected error: {str(e)}")

def main():
    searcher = APISearch()
    asyncio.run(searcher.search_loop())

if __name__ == "__main__":
    main()
