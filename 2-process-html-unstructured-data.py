#!/usr/bin/env python3

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import argparse

from bs4 import BeautifulSoup
import spacy
from spacy.language import Language

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndexEntry:
    """Data class for storing summary data"""
    def __init__(self, url: str, title: str, summary: str):
        self.url = url
        self.title = title
        self.summary = summary

class TextSummarizer:
    def __init__(self, output_dir: str = "data"):
        """Initialize the text summarizer with spaCy NLP"""
        self.output_dir = Path(output_dir)
        self.nlp = self._initialize_spacy()
        logger.info("Initialized TextSummarizer")

    def _initialize_spacy(self) -> Language:
        """Initialize spaCy with error handling"""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model")
            return nlp
        except OSError:
            logger.warning("SpaCy model 'en_core_web_sm' not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text, removing unwanted words for RAG retrieval"""
        # Remove irrelevant terms for RAG (e.g., 'menu', 'html', 'title')
        text = re.sub(r'\b(menu|html|title|include)\b', '', text, flags=re.IGNORECASE)

        # Remove special characters and multiple spaces
        text = re.sub(r'[^\w\s-]', ' ', text)  # Remove special characters
        text = re.sub(r'-+', ' ', text)        # Replace dashes with spaces
        text = re.sub(r'\s+', ' ', text)       # Collapse multiple spaces

        return text.strip().lower()

    def summarize_text(self, text: str) -> str:
        """Summarize the text using spaCy"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        # Return the first 3 sentences as a simple summary (can be adjusted)
        return ' '.join(sentences[:3])

    def process_html_file(self, file_path: Path) -> Optional[IndexEntry]:
        """Process a single HTML file and return a summary"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')

                # Remove script and style elements
                for element in soup(['script', 'style']):
                    element.decompose()

                # Extract and clean text
                body_text = soup.get_text(separator=' ', strip=True)
                clean_body_text = self.clean_text(body_text)

                # Skip if no meaningful content
                if not clean_body_text:
                    logger.warning(f"Skipping {file_path}: No meaningful content")
                    return None

                # Generate summary
                summary = self.summarize_text(clean_body_text)

                # Create URL path and ensure it's valid
                relative_path = str(file_path.relative_to(Path.cwd()))
                url_path = f"https://kevinluzbetak.com/{relative_path}"

                return IndexEntry(
                    url=url_path.strip(),
                    title=file_path.name,
                    summary=summary
                )

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def _write_index_file(self, entries: List[IndexEntry]) -> None:
        """Write the summaries to a JSON file"""
        self.output_dir.mkdir(exist_ok=True)
        output_file = self.output_dir / "search-index.json"

        # Final validation
        valid_entries = [
            vars(entry) for entry in entries
            if entry.url and entry.title and entry.summary
        ]

        if not valid_entries:
            logger.error("No valid entries to write!")
            return

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(valid_entries, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully created {output_file} with {len(valid_entries)} entries")
        except Exception as e:
            logger.error(f"Error writing index file: {e}")
            raise

    def generate_index(self) -> None:
        """Generate the summary index from HTML files"""
        logger.info("Starting summary index generation...")

        # Collect HTML files
        html_files = [
            p for p in Path.cwd().rglob("*.html")
            if p.name != "index.html" and not str(p).startswith(str(self.output_dir))
        ]

        if not html_files:
            logger.warning("No HTML files found to process")
            return

        logger.info(f"Found {len(html_files)} HTML files to process")

        # Process files in parallel
        with ThreadPoolExecutor() as executor:
            entries = list(filter(None, executor.map(
                self.process_html_file,
                html_files
            )))

        if not entries:
            logger.error("No valid entries generated from HTML files")
            return

        # Write summaries to JSON file
        self._write_index_file(entries)

        logger.info("Summary index generation completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate summary index from HTML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for the summary index"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        summarizer = TextSummarizer(args.output_dir)
        summarizer.generate_index()
    except Exception as e:
        logger.error(f"Failed to generate summary index: {e}")
        exit(1)

if __name__ == "__main__":
    main()

