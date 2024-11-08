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

class CLISearch:
    def __init__(self):
        self.query_engine = QueryEngine()
        # Initialize summarizer options
        self.initialize_summarizers()
        
    def initialize_summarizers(self):
        """Initialize different summarization models"""
        # HuggingFace transformer-based summarizer
        self.hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # SUMY LSA summarizer
        self.lsa_summarizer = LsaSummarizer(Stemmer('english'))
        self.lsa_summarizer.stop_words = get_stop_words('english')
        
        # ROUGE metric for evaluation
        self.rouge = Rouge()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def summarize_with_transformers(self, text, max_length=130, min_length=30):
        """
        Use BART model for abstractive summarization
        """
        try:
            summary = self.hf_summarizer(text, 
                                       max_length=max_length, 
                                       min_length=min_length, 
                                       do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logger.error(f"Transformer summarization failed: {e}")
            return None

    def summarize_with_sumy(self, text, sentences_count=3):
        """
        Use SUMY's LSA for extractive summarization
        """
        try:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summary = self.lsa_summarizer(parser.document, sentences_count)
            return ' '.join([str(sentence) for sentence in summary])
        except Exception as e:
            logger.error(f"SUMY summarization failed: {e}")
            return None

    def get_best_summary(self, text, original_query):
        """
        Generate and evaluate summaries using different methods
        """
        summaries = {}
        
        # Generate summaries using different methods
        transformer_summary = self.summarize_with_transformers(text)
        if transformer_summary:
            summaries['transformer'] = transformer_summary
            
        sumy_summary = self.summarize_with_sumy(text)
        if sumy_summary:
            summaries['sumy'] = sumy_summary
            
        if not summaries:
            return self.fallback_summarization(text)
            
        # Evaluate summaries using ROUGE and query relevance
        best_score = -1
        best_summary = None
        
        for method, summary in summaries.items():
            # Calculate ROUGE scores
            try:
                scores = self.rouge.get_scores(summary, text)[0]
                rouge_score = (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3
                
                # Calculate query term overlap
                query_terms = set(original_query.lower().split())
                summary_terms = set(summary.lower().split())
                query_overlap = len(query_terms.intersection(summary_terms)) / len(query_terms)
                
                # Combined score with weights
                total_score = (0.7 * rouge_score) + (0.3 * query_overlap)
                
                if total_score > best_score:
                    best_score = total_score
                    best_summary = summary
                    
            except Exception as e:
                logger.error(f"Error evaluating {method} summary: {e}")
                continue
                
        return best_summary or self.fallback_summarization(text)

    def fallback_summarization(self, text, max_sentences=3):
        """
        Simple extractive summarization as fallback
        """
        sentences = nltk.sent_tokenize(text)
        return ' '.join(sentences[:max_sentences])

    def generate_concise_answer(self, results, query):
        """
        Generate concise answer using advanced summarization
        """
        if not results:
            return "No relevant information found."

        # Combine relevant content from results
        combined_text = ' '.join([doc.get('content', '').strip() 
                                for doc in sorted(results, 
                                                key=lambda x: x.get('score', 0), 
                                                reverse=True)])
        
        # Generate summary
        summary = self.get_best_summary(combined_text, query)
        
        if not summary:
            return "Could not generate a summary from the search results."
            
        return summary

    def print_results(self, results, query):
        print("\nGenerated Answer:")
        print("-" * 50)
        print(self.generate_concise_answer(results, query))

    async def search_loop(self):
        print("\nRAG CLI Search")
        print("=" * 50)
        print("Enter 'exit' to quit")
        print("=" * 50)

        while True:
            query = input("\nEnter search query: ")
            if query.lower() == 'exit':
                break

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
