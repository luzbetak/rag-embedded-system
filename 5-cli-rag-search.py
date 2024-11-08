#!/usr/bin/env python3

import os
import asyncio
from query import QueryEngine
from loguru import logger
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from rouge import Rouge
import re

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()
        self.initialize_summarizers()
        
    def initialize_summarizers(self):
        """Initialize different summarization models"""
        try:
            self.hf_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1
            )
        except Exception as e:
            logger.error(f"Failed to initialize transformer model: {e}")
            self.hf_summarizer = None
            
        self.lsa_summarizer = LsaSummarizer(Stemmer('english'))
        self.lsa_summarizer.stop_words = get_stop_words('english')
        self.rouge = Rouge()
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def preprocess_text(self, text):
        """
        Preprocess text to make it more manageable for summarization
        """
        if not text:
            return ""
            
        # Remove duplicate whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove duplicate sentences (common in web scrapes)
        sentences = nltk.sent_tokenize(text)
        unique_sentences = list(dict.fromkeys(sentences))
        
        # Limit to first 5 sentences for initial summary
        limited_text = ' '.join(unique_sentences[:5])
        
        return limited_text.strip()

    def summarize_with_transformers(self, text, max_length=130, min_length=30):
        """
        Use BART model for abstractive summarization
        """

        # For longer summaries:
        max_length = 250  # Allow longer summaries
        min_length = 50   # Ensure more detailed minimum content

        # For shorter summaries:
        max_length = 75   # Keep summaries very concise
        min_length = 15   # Allow very brief summaries

        # For medium summaries:
        max_length = 100  # Limits summary to ~100 words/tokens - good for paragraph-length summaries that capture main points
        min_length = 20   # Ensures summary is at least ~20 words/tokens - prevents overly short/incomplete summaries

        if self.hf_summarizer is None:
            return None

        try:
            # Preprocess and limit text
            text = self.preprocess_text(text)
            
            # If text is too short, return it as is
            if len(text.split()) < 50:  # Reduced minimum length
                return text

            # Generate summary with conservative length limits
            summary = self.hf_summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            if summary and len(summary) > 0:
                return summary[0]['summary_text']
            return None

        except Exception as e:
            logger.error(f"Transformer summarization failed: {e}")
            return None

    def summarize_with_sumy(self, text, sentences_count=2):  # Reduced sentence count
        """
        Use SUMY's LSA for extractive summarization
        """
        try:
            text = self.preprocess_text(text)
            if not text:
                return None
                
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary = self.lsa_summarizer(parser.document, sentences_count)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            logger.error(f"SUMY summarization failed: {e}")
            return None

    def fallback_summarization(self, text, max_sentences=2):  # Reduced sentence count
        """Simple extractive summarization as fallback"""
        try:
            text = self.preprocess_text(text)
            if not text:
                return None
                
            sentences = nltk.sent_tokenize(text)
            return ' '.join(sentences[:max_sentences])
        except Exception as e:
            logger.error(f"Fallback summarization failed: {e}")
            return text[:200] + "..."  # Last resort: just truncate

    def get_best_summary(self, text, original_query):
        """Generate and evaluate summaries"""
        if not text or not original_query:
            return "No text to summarize."

        # Try transformer first
        summary = self.summarize_with_transformers(text)
        if summary:
            return summary

        # Try SUMY if transformer fails
        summary = self.summarize_with_sumy(text)
        if summary:
            return summary

        # Use fallback if both fail
        return self.fallback_summarization(text)

    def generate_concise_answer(self, results, query):
        """Generate concise answer from search results"""
        if not results:
            return "No relevant information found."

        try:
            # Take only top 3 results
            top_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)[:3]
            
            # Combine content with length limit
            combined_text = ' '.join([doc.get('content', '').strip()[:500] 
                                    for doc in top_results])
            
            summary = self.get_best_summary(combined_text, query)
            return summary if summary else "Could not generate a summary."
            
        except Exception as e:
            logger.error(f"Error generating concise answer: {e}")
            return "Error generating summary."

    def print_results(self, results, query):
        """Print formatted results"""
        print("\nGenerated Answer:")
        print("-" * 50)
        print(self.generate_concise_answer(results, query))

    async def search_loop(self):
        """Main search loop"""
        print("\nRAG CLI Search")
        print("=" * 50)
        print("Enter 'exit' to quit")
        print("=" * 50)

        while True:
            try:
                query = input("\nEnter search query: ").strip()
                if query.lower() == 'exit':
                    break

                if not query:
                    print("Please enter a valid query.")
                    continue

                results = await self.query_engine.search(query)
                self.print_results(results, query)
                
            except Exception as e:
                logger.error(f"Error during search: {e}")
                print(f"\nError during search: {str(e)}")

def main():
    try:
        searcher = CLISearch()
        asyncio.run(searcher.search_loop())
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error occurred: {str(e)}")

if __name__ == "__main__":
    main()
