#!/usr/bin/env python3
import asyncio
import aiohttp
from loguru import logger
import os

class RAGSearchInterface:
    def __init__(self):
        self.api_url = "http://localhost:8000/search"
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        print("=" * 50)
        print("RAG Search System")
        print("=" * 50)
        print("Options:")
        print("1. Search documents (CLI)")
        print("2. Search using API server")
        print("3. Exit")
        print("=" * 50)
        
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
            
    async def search_documents_cli(self):
        from query import QueryEngine
        query_engine = QueryEngine()
        
        query = input("\nEnter your search query: ")
        print("\nSearching using CLI...")
        
        try:
            results = await query_engine.search(query)
            self.print_results(results)
        except Exception as e:
            print(f"\nError during search: {str(e)}")
            
        input("\nPress Enter to continue...")
    
    async def search_documents_api(self):
        query = input("\nEnter your search query: ")
        print("\nSearching using API...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json={"text": query, "top_k": 5}
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
            
        input("\nPress Enter to continue...")
        
    async def main_loop(self):
        while True:
            self.clear_screen()
            self.print_header()
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                await self.search_documents_cli()
            elif choice == '2':
                await self.search_documents_api()
            elif choice == '3':
                print("\nGoodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")
                input("\nPress Enter to continue...")

def main():
    interface = RAGSearchInterface()
    asyncio.run(interface.main_loop())

if __name__ == "__main__":
    main()
