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
    def __init__(self):
        """Initialize the document validator with required settings"""
        self.required_fields = ['url', 'title', 'content']
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
        content = ' '.join(content.split())
        content = re.sub(r'[^\w\s]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
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

            if not cleaned_url:
                logger.warning(f"Invalid URL in document: {doc.get('title', 'Unknown')}")
                return None

            if not cleaned_content:
                logger.warning(f"Empty content after cleaning in document: {doc.get('title', 'Unknown')}")
                return None

            validated_doc = {
                "url": cleaned_url,
                "title": cleaned_title or "Untitled",
                "content": cleaned_content
            }

            return validated_doc

        except Exception as e:
            logger.error(f"Error validating document: {str(e)}")
            return None

    def batch_validate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of documents"""
        valid_docs = []
        total_docs = len(documents)
        skipped_docs = 0

        logger.info(f"Starting validation of {total_docs} documents")

        for index, doc in enumerate(documents, 1):
            validated_doc = self.validate_document(doc)
            if validated_doc:
                valid_docs.append(validated_doc)
            else:
                skipped_docs += 1

            if index % 100 == 0:
                logger.info(f"Processed {index}/{total_docs} documents")

        logger.info(f"Validation complete: {len(valid_docs)} valid, {skipped_docs} skipped")
        return valid_docs

    def display_summary(self, documents: List[Dict[str, Any]]) -> None:
        """Display a summary of the validated documents"""
        # Create a table for document summary
        table = Table(title="Document Validation Summary")
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Value", justify="left", style="green")

        # Add summary statistics
        table.add_row("Total Documents", str(len(documents)))
        table.add_row("Unique URLs", str(len(set(doc['url'] for doc in documents))))
        table.add_row("Average Content Length", 
                     str(round(sum(len(doc['content'].split()) for doc in documents) / len(documents))))

        console.print("\n")
        console.print(Panel("[bold blue]Document Validation Results[/bold blue]"))
        console.print(table)

        # Display sample documents
        console.print("\n[bold]Sample Documents:[/bold]")
        for i, doc in enumerate(documents[:3], 1):
            console.print(f"\n[cyan]Document {i}:[/cyan]")
            console.print(f"Title: {doc['title']}")
            console.print(f"URL: {doc['url']}")
            content_preview = ' '.join(doc['content'].split()[:10]) + '...'
            console.print(f"Content Preview: {content_preview}")

    def validate_file(self, input_file: str, output_file: Optional[str] = None, display: bool = True) -> None:
        """Validate documents from a JSON file and optionally save results"""
        try:
            input_path = Path(input_file)
            if not input_path.exists():
                logger.error(f"Input file not found: {input_file}")
                return

            logger.info(f"Reading documents from {input_file}")
            with open(input_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            valid_documents = self.batch_validate_documents(documents)

            if not valid_documents:
                logger.error("No valid documents found after validation")
                return

            if display:
                self.display_summary(valid_documents)

            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(valid_documents, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(valid_documents)} validated documents to {output_file}")

            return valid_documents

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in {input_file}")
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")

def main():
    """Main entry point for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate documents for RAG system")
    parser.add_argument("input_file", help="Input JSON file containing documents")
    parser.add_argument("--output", "-o", help="Output file for validated documents")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-display", action="store_true", help="Disable summary display")

    args = parser.parse_args()

    if args.debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")

    validator = DocumentValidator()
    validator.validate_file(args.input_file, args.output, display=not args.no_display)

if __name__ == "__main__":
    main()

