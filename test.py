#!/usr/bin/env python3
import os
import asyncio
from query import QueryEngine
from loguru import logger
import nltk
import re

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()
        self.download_nltk_data()

    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}. Using fallback tokenization.")

    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove code snippets and imports
        text = re.sub(r'import\s+\w+(?:\s*,\s*\w+)*', '', text)
        text = re.sub(r'from\s+\w+(?:\.[.\w]+)?\s+import\s+(?:\w+(?:\s*,\s*\w+)*)', '', text)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove file paths and commands
        text = re.sub(r'(?:/[\w.-]+)+', '', text)
        text = re.sub(r'(?:usr/bin/env)\s+\w+', '', text)
        
        # Basic cleaning while preserving sentence structure
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()

    def extract_key_sentence(self, text, query):
        """Extract the most relevant sentence containing query terms"""
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        query_terms = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            # Skip very short sentences or those that look like code
            if len(sentence.split()) < 5 or 'import' in sentence or 'def ' in sentence:
                continue
                
            sentence_terms = set(sentence.lower().split())
            relevance_score = len(query_terms.intersection(sentence_terms))
            
            # Prefer sentences of moderate length
            length_score = min(1.0, 30.0 / len(sentence_terms))
            final_score = relevance_score * length_score
            
            if final_score > 0:
                relevant_sentences.append((final_score, sentence))
        
        if relevant_sentences:
            relevant_sentences.sort(reverse=True)
            return relevant_sentences[0][1]
        elif sentences:
            # Find the first non-code, reasonable length sentence
            for sentence in sentences:
                if len(sentence.split()) >= 5 and 'import' not in sentence and 'def ' not in sentence:
                    return sentence
        return ""

    def generate_answer(self, results, query):
        """Generate a very concise answer from search results"""
        if not results:
            return "No relevant information found."

        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        # Extract metadata from top result for context
        top_doc = sorted_results[0]
        title = top_doc.get('metadata', {}).get('title', '')
        
        # Get key sentences from top 2 documents
        key_sentences = []
        for doc in sorted_results[:2]:
            sentence = self.extract_key_sentence(self.clean_text(doc.get('content', '')), query)
            if sentence and sentence not in key_sentences:
                key_sentences.append(sentence)
        
        if not key_sentences:
            return "Could not extract relevant information from the search results."
        
        # Include title in response if available
        context = f"From '{title}': " if title else ""
        
        if len(key_sentences) > 1:
            return f"{context}{key_sentences[0]}. Additionally, {key_sentences[1].lower()}"
        else:
            return f"{context}{key_sentences[0]}"

    def print_results(self, results, query):
        """Print search results with complete metadata"""
        if not results:
            print("\nNo results found.")
            return

        print("\nGenerated Answer:")
        print("-" * 50)
        print(self.generate_answer(results, query))
        
        print("\nSource Documents:")
        print("-" * 50)
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i}:")
            
            metadata = doc.get('metadata', {})
            
            # Display title if available
            title = metadata.get('title', '')
            if title:
                print(f"Title: {title}")
            
            # Display URL/source path
            url = metadata.get('url', '')
            source = metadata.get('source', '')
            if url:
                print(f"URL: {url}")
            elif source:
                print(f"Source: {source}")
            else:
                print("Source: N/A")
            
            # Display relevance score
            if 'score' in doc:
                print(f"Relevance Score: {doc['score']:.3f}")

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
                
            if not query.strip():
                print("Please enter a valid query.")
                continue

            print("\nSearching...")
            try:
                results = await self.query_engine.search(query)
                self.print_results(results, query)
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                print(f"\nError during search: {str(e)}")

def main():
    try:
        searcher = CLISearch()
        asyncio.run(searcher.search_loop())
    except KeyboardInterrupt:
        print("\nSearch terminated by user.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
