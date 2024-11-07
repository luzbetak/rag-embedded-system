#!/usr/bin/env python3
import os
import asyncio
from query import QueryEngine
from loguru import logger

# Set OpenBLAS environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE']   = '1'
os.environ['OMP_NUM_THREADS']      = '1'

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()

    def generate_concise_answer(self, results, query):
        """
        Combines search results into a brief, focused answer.
        
        Args:
            results (list): List of search result documents
            query (str): Original search query
            
        Returns:
            str: Concise RAG Answer Generator 
        """
        if not results:
            return "No relevant information found."

        # Sort results by similarity score
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        # Extract the most relevant sentences from each result
        key_points = []
        for doc in sorted_results:
            content = doc.get('content', '').strip()
            # Split into sentences and take the most relevant ones
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            # Take only first relevant sentence containing the query terms
            for sentence in sentences:
                if any(term.lower() in sentence.lower() for term in query.split()):
                    key_points.append(sentence)
                    break

        # Limit to top 3 key points
        key_points = key_points[:3]
        
        if not key_points:
            return "Found results but couldn't extract relevant points for your query."

        # Combine key points into a concise answer
        if len(key_points) == 1:
            return f"{key_points[0]}."
        
        answer = f"{key_points[0]}. "
        if len(key_points) >= 2:
            answer += f"Additionally, {key_points[1].lower()}"
        if len(key_points) == 3:
            answer += f". Finally, {key_points[2].lower()}"
        
        return answer + "."

    def print_results(self, results, query):
        print("\nSearch Results:")
        print("-" * 50)
        if not results:
            print("\nNo documents found.")
            return

        # Generate and print the concise answer
        print("\nGenerated Answer:")
        print("-" * 50)
        print(self.generate_concise_answer(results, query))
        
        # Optionally print detailed results
        print("\nDetailed Results:")
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"Title: {doc.get('title', 'N/A')}")
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
                self.print_results(results, query)
            except Exception as e:
                print(f"\nError during search: {str(e)}")

def main():
    searcher = CLISearch()
    asyncio.run(searcher.search_loop())

if __name__ == "__main__":
    main()
