#!/usr/bin/env python3

import os
import json
import re
import logging
from pathlib import Path
from typing import Set, List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import argparse

from bs4 import BeautifulSoup
import spacy
from spacy.language import Language
import nltk
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IndexEntry:
    """Data class for search index entries"""
    url: str       # Primary identifier
    title: str     # HTML file title
    content: str   # Processed content for search

class SearchIndexGenerator:
    def __init__(self, output_dir: str = "data"):
        """Initialize the search index generator with necessary NLP components"""
        self.output_dir = Path(output_dir)
        self.nlp = self._initialize_spacy()
        self.stop_words = self._initialize_nltk()
        logger.info("Initialized SearchIndexGenerator")

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

    def _initialize_nltk(self) -> Set[str]:
        """Initialize NLTK components"""
        try:
            nltk.download('stopwords', quiet=True)
            stop_words = set(stopwords.words('english'))
            logger.info("Loaded NLTK stopwords")
            return stop_words
        except Exception as e:
            logger.error(f"Error initializing NLTK: {e}")
            raise

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove special characters and excess whitespace
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'-+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and normalize title"""
        # Remove file extension and replace separators with spaces
        title = title.replace('.html', '')
        title = re.sub(r'[-_]', ' ', title)
        return title.strip()

    def extract_technical_terms(self, text: str) -> Set[str]:
        """Extract technical terms using spaCy NLP"""
        doc = self.nlp(text)
        technical_terms = set()

        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in {'ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'EVENT'}:
                technical_terms.add(ent.text.lower())

        # Extract noun chunks and filter out stopwords
        for chunk in doc.noun_chunks:
            if not all(token.text.lower() in self.stop_words for token in chunk):
                technical_terms.add(chunk.text.lower())

        # Add important individual tokens
        for token in doc:
            if (token.pos_ in {'NOUN', 'PROPN'} and 
                token.text.lower() not in self.stop_words and
                len(token.text) > 2):
                technical_terms.add(token.text.lower())

        return technical_terms

    def process_html_file(self, file_path: Path) -> Optional[IndexEntry]:
        """Process a single HTML file and return an index entry"""
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

                # Create URL path and ensure it's valid
                try:
                    relative_path = str(file_path.relative_to(Path.cwd()))
                    url_path = f"https://kevinluzbetak.com/{relative_path}"
                    
                    if not url_path or url_path.isspace():
                        logger.warning(f"Skipping {file_path}: Invalid URL path")
                        return None

                    # Extract technical terms
                    technical_terms = self.extract_technical_terms(clean_body_text)
                    
                    if not technical_terms:
                        logger.warning(f"Skipping {file_path}: No technical terms extracted")
                        return None

                    return IndexEntry(
                        url=url_path.strip(),
                        title=file_path.name,
                        content=" ".join(sorted(technical_terms))
                    )
                except ValueError as e:
                    logger.warning(f"Skipping {file_path}: Unable to create valid URL path")
                    return None

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def _post_process_entries(self, entries: List[IndexEntry]) -> None:
        """Post-process entries to ensure unique content and clean titles"""
        unique_entries = {}
        
        for entry in entries:
            # Skip invalid entries
            if not entry.url or not entry.title:
                continue

            # Clean title
            cleaned_title = self.clean_title(entry.title)
            if not cleaned_title:
                logger.warning(f"Skipping entry with empty title after cleaning: {entry.url}")
                continue

            # Process content
            words = entry.content.split()
            filtered_words = [
                word.lower() for word in words
                if word.isalpha() and word.lower() not in self.stop_words
            ]

            if not filtered_words:
                logger.warning(f"Skipping entry with no valid content after filtering: {entry.url}")
                continue

            # Create final processed entry
            unique_words = sorted(set(filtered_words))
            processed_content = f"{cleaned_title.lower()} {' '.join(unique_words)}"

            # Store using URL as key
            unique_entries[entry.url] = {
                "url": entry.url,
                "title": entry.title,
                "content": processed_content
            }

        # Update entries list
        entries.clear()
        entries.extend([IndexEntry(**entry_data) for entry_data in unique_entries.values()])
        logger.info(f"Post-processing complete: {len(entries)} valid entries")

    def _write_index_file(self, entries: List[IndexEntry]) -> None:
        """Write the search index to a JSON file"""
        self.output_dir.mkdir(exist_ok=True)
        output_file = self.output_dir / "search-index.json"

        # Final validation
        valid_entries = [
            vars(entry) for entry in entries 
            if entry.url and entry.title and entry.content
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
        """Generate the search index from HTML files"""
        logger.info("Starting search index generation...")

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

        # Post-process entries
        self._post_process_entries(entries)

        # Write index file
        self._write_index_file(entries)
        
        logger.info("Search index generation completed")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate search index from HTML files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for search index"
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
        generator = SearchIndexGenerator(args.output_dir)
        generator.generate_index()
    except Exception as e:
        logger.error(f"Failed to generate search index: {e}")
        exit(1)

if __name__ == "__main__":
    main()
