#!/usr/bin/env python3

import re
from typing import Dict, Any, Optional, List
from loguru import logger
import json
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class DocumentValidator:
    def __init__(self, default_input='data/search-index.json', default_output='data/validated-index.json'):
        """Initialize the document validator with default settings"""
        self.required_fields = ['url', 'title', 'content']
        self.default_input = default_input
        self.default_output = default_output
        logger.info("Initialized DocumentValidator")

    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and validate URL"""
        if not url:
            return ''
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        return url

    @staticmethod
    def clean_title(title: str) -> str:
        """Clean and normalize title"""
        if not title:
            return ''
        title = ' '.join(title.split())
        return title.strip()

    @staticmethod
    def clean_content(content: str) -> str:
        """Clean and normalize content"""
        if not content:
            return ''
        # Remove special characters but keep periods and commas
        content = re.sub(r'[^\w\s.,]', ' ', content)
        content = ' '.join(content.split())
        return content.strip().lower()

    def validate_document(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and clean individual document fields"""
        try:
            if not doc:
                logger.warning("Empty document received")
                return None

            missing_fields = [field for field in self.required_fields if field not in doc]
            if missing_fields:
                logger.warning(f"Document missing required fields: {missing_fields}")
                return None

            cleaned_url = self.clean_url(doc['url'])
            cleaned_title = self.clean_title(doc['title'])
            cleaned_content = self.clean_content(doc['content'])

            if not cleaned_url or not re.match(r'^https?://', cleaned_url):
                logger.warning(f"Invalid URL in document: {doc.get('title', 'Unknown')}")
                return None

            if len(cleaned_content.split()) < 10:  # Minimum content length
                logger.warning(f"Content too short in document: {doc.get('title', 'Unknown')}")
                return None

            validated_doc = {
                "url": cleaned_url,
                "title": cleaned_title or "Untitled",
                "content": cleaned_content,
                "metadata": {
                    "word_count": len(cleaned_content.split()),
                    "original_length": len(doc.get('content', '')),
                    "cleaned_length": len(cleaned_content)
                }
            }

            return validated_doc

        except Exception as e:
            logger.error(f"Error validating document: {str(e)}")
            return None

    def batch_validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of documents with progress tracking"""
        valid_docs = []
        total_docs = len(documents)
        skipped_docs = 0

        with console.status("[bold green]Validating documents...") as status:
            for index, doc in enumerate(documents, 1):
                validated_doc = self.validate_document(doc)
                if validated_doc:
                    valid_docs.append(validated_doc)
                else:
                    skipped_docs += 1

                if index % 50 == 0:
                    status.update(f"[bold green]Processed {index}/{total_docs} documents")
                    logger.info(f"Processed {index}/{total_docs} documents")

        logger.info(f"Validation complete: {len(valid_docs)} valid, {skipped_docs} skipped")
        return valid_docs

    def display_summary(self, documents: List[Dict[str, Any]]) -> None:
        """Display an enhanced summary of the validated documents"""
        table = Table(title="Document Validation Summary")
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="left", style="green")

        # Enhanced statistics
        total_docs = len(documents)
        avg_word_count = sum(doc['metadata']['word_count'] for doc in documents) / total_docs if total_docs else 0
        avg_reduction = sum((doc['metadata']['original_length'] - doc['metadata']['cleaned_length']) / 
                          doc['metadata']['original_length'] * 100 for doc in documents) / total_docs if total_docs else 0

        stats = [
            ("Total Documents", str(total_docs)),
            ("Unique URLs", str(len(set(doc['url'] for doc in documents)))),
            ("Average Word Count", f"{avg_word_count:.1f}"),
            ("Average Content Reduction", f"{avg_reduction:.1f}%"),
            ("Shortest Document", str(min(doc['metadata']['word_count'] for doc in documents))),
            ("Longest Document", str(max(doc['metadata']['word_count'] for doc in documents)))
        ]

        for metric, value in stats:
            table.add_row(metric, value)

        console.print("\n")
        console.print(Panel("[bold blue]Document Validation Results[/bold blue]"))
        console.print(table)

        # Sample preview
        if documents:
            console.print("\n[bold]Sample Document Preview:[/bold]")
            doc = documents[0]
            console.print(Panel(
                f"[cyan]Title:[/cyan] {doc['title']}\n"
                f"[cyan]URL:[/cyan] {doc['url']}\n"
                f"[cyan]Content Preview:[/cyan] {' '.join(doc['content'].split()[:20])}...\n"
                f"[cyan]Word Count:[/cyan] {doc['metadata']['word_count']}"
            ))

    def validate_file(self, input_file: Optional[str] = None, output_file: Optional[str] = None, display: bool = True) -> None:
        """Validate documents from a JSON file with default paths"""
        try:
            input_path = Path(input_file or self.default_input)
            output_path = Path(output_file or self.default_output)

            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return

            logger.info(f"Reading documents from {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            valid_documents = self.batch_validate_documents(documents)

            if not valid_documents:
                logger.error("No valid documents found after validation")
                return

            if display:
                self.display_summary(valid_documents)

            # Always save validated documents
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(valid_documents, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(valid_documents)} validated documents to {output_path}")

            return valid_documents

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {input_path}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")

def main():
    """Main entry point with default file paths"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate documents for RAG system")
    parser.add_argument("--input", "-i", help="Input JSON file (default: data/search-index.json)")
    parser.add_argument("--output", "-o", help="Output file (default: data/validated-index.json)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-display", action="store_true", help="Disable summary display")

    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")

    validator = DocumentValidator()
    validator.validate_file(args.input, args.output, display=not args.no_display)

if __name__ == "__main__":
    main()
