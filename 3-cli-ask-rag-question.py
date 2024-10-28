#!/usr/bin/env python3

import asyncio
from query import QueryEngine
from loguru import logger
import os

class RAGSearchInterface:
    def __init__(self):
        self.query_engine = QueryEngine()
        
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def print_header(self):
        print("=" * 50)
        print("RAG Search System")
        print("=" * 50)
        print("Options:")
        print("1. Search documents")
        print("2. Exit")
        print("=" * 50)
        
    def print_results2(self, results):
        print("\nSearch Results:")
        print("-" * 50)
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"Title: {doc.get('title', 'N/A')}")
            print(f"Content: {doc.get('content', 'N/A')}")
            if 'metadata' in doc:
                print("Metadata:")
                for key, value in doc['metadata'].items():
                    print(f"  {key}: {value}")
            print(f"Similarity Score: {doc.get('score', 'N/A')}")
            print("-" * 30)

    def print_results(self, results):
        print("\nSearch Results:")
        print("-" * 50)
    
        # Remove duplicates based on title and content
        seen = set()
        unique_results = []
        for doc in results:
            key = (doc.get('title', ''), doc.get('content', ''))
            if key not in seen:
                seen.add(key)
                unique_results.append(doc)
    
        if not unique_results:
            print("\nNo documents found.")
            return
    
        for i, doc in enumerate(unique_results, 1):
            print(f"\nDocument {i}:")
            print(f"Title: {doc.get('title', 'N/A')}")
            print(f"Content: {doc.get('content', 'N/A')}")
            if doc.get('metadata'):
                print("Metadata:")
                for key, value in doc['metadata'].items():
                    print(f"  {key}: {value}")
            if 'score' in doc and doc['score'] > 0:
                print(f"Similarity Score: {doc['score']:.3f}")
            print("-" * 30)

    async def search_documents(self):
        query = input("\nEnter your search query: ")
        print("\nSearching...")
        
        try:
            results = await self.query_engine.search(query)
            self.print_results(results)
        except Exception as e:
            print(f"\nError during search: {str(e)}")
            
        input("\nPress Enter to continue...")
        
    async def main_loop(self):
        while True:
            self.clear_screen()
            self.print_header()
            
            choice = input("\nEnter your choice (1-2): ")
            
            if choice == '1':
                await self.search_documents()
            elif choice == '2':
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
