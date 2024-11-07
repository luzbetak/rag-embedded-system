Retrieval-Augmented Generation (RAG) by Kevin Luzbetak
======================================================

RAG system using MongoDB for document storage and sentence-transformers for embeddings.

--------------------------------------------------------------------------------------------
## Quick Start

```bash
# Process HTML files
python 1-process-summarize-unstructured-html.py -s textrank

# Initialize database (delete existing data)
# Load documents from data/search-index.json
./2-document-summarize-validator.py

# Direct CLI search
./4-cli-rag-search.py 
```

--------------------------------------------------------------------------------------------
## Files

```bash
┌── 1-process-summarize-unstructured-html.py # Process HTML files and generate index
├── 2-document-summarize-validator.py        # Validate and summarize documents
├── 3-initialize-db-load-documents.py        # Initialize MongoDB and store documents
├── 4-cli-rag-search.py                      # Direct CLI search tool
├── 5-run-fastapi-uvicorn-server.sh          # Start FastAPI server
└── 6-api-rag-search.py                      # API-based search tool

┌── ai/                                      # Directory with unstructured HTML files
├── config.py                                # System configuration
├── database.py                              # MongoDB database interactions
├── 6ata_ingestion.py                        # Data loading and preprocessing pipeline
├── vectorization.py                         # Handles document embedding
└── query.py                                 # Search and retrieval engine
```
--------------------------------------------------------------------------------------------
## Default Embedding Model:
The all-MiniLM-L6-v2 is a lightweight and efficient transformer model that's popular for generating sentence embeddings. Here are its key characteristics:

## Architecture all-MiniLM-L6-v2:
Based on MiniLM architecture, which is a distilled version of larger transformer models
6 layers (L6 in the name) making it quite compact
Trained using knowledge distillation from larger models

### Main strengths:
Very fast inference speed due to its small size
Good balance of performance vs computational requirements
Produces 384-dimensional embeddings
Works well for general-purpose sentence similarity tasks

### Compared to alternatives listed:
Faster but slightly lower quality than all-mpnet-base-v2
More general-purpose than multi-qa-MiniLM-L6-cos-v1 which is QA-specific
English-only, unlike the multilingual variant
Similar performance tier to all-distilroberta-v1 but typically faster

### Best suited for:
Production environments where speed is important
Applications with resource constraints
General semantic similarity tasks
Cases where a good speed/performance trade-off is needed

It's often considered a good default choice when you need reliable embeddings without excessive computational overhead. The model strikes a nice balance between efficiency and effectiveness for most common use cases.

### Alternative Embedding Models:
- sentence-transformers/all-MiniLM-L6-v2
- sentence-transformers/all-mpnet-base-v2                      # Better quality, slower
- sentence-transformers/multi-qa-MiniLM-L6-cos-v1              # Optimized for QA
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  # Multilingual
- sentence-transformers/all-distilroberta-v1                   # Good balance of speed/quality

Summarization: TextRank (default), spaCy, BART


## Additional Notes:
- The script uses sentence-transformers for generating embeddings
- Default model: 'sentence-transformers/all-MiniLM-L6-v2'
- Embedding dimension: 384
- MongoDB indices optimize search performance
- Logging is handled by loguru for better debugging


-----------------------------------------------------------------------------------------------------
## Process Unstructured HTML and Summarization
```bash
pip install spacy networkx numpy beautifulsoup4
python -m spacy download en_core_web_sm

# Run with TextRank (default)
python 2-process-unstructured-html.py

# Run with basic summarization
python 2-process-unstructured-html.py --summarize basic 
python 2-process-unstructured-html.py --summarize textrank

# Run with debug logging
python 2-process-unstructured-html.py --debug
```
-----------------------------------------------------------------------------------------------------
## Document Summarization Validation
```bash
python -m spacy download en_core_web_sm

# Basic usage
python document_validator.py

# With specific input/output files
python document_validator.py -i input.json -o output.json

# Debug mode
python document_validator.py --debug

# Basic summarization (default)
python document_validator.py


# Using different summarization methods
python document_validator.py --summarize textrank
python document_validator.py --summarize spacy
python document_validator.py --summarize transformers

```
-----------------------------------------------------------------------------------------------------
![Word Similarity Heatmap](embeddings/similarity_heatmap.png)


```bash
+-----------+------------------------------------------------------------------------------------------------------+
| Field     | Value                                                                                                |
+-----------+------------------------------------------------------------------------------------------------------+
| _id       | 672d49d34d10561a9cda098e                                                                             |
| url       | https://luzbetak.github.io/ai/python-whoosh.html                                                     |
| title     | python-whoosh.html                                                                                   |
| content   | python whoosh create python bm25 index usr bin env python from whoosh index import create_in from    |
|           | whoosh fields import schema text id from whoosh import qparser from whoosh qparser import            |
|           | queryparser from whoosh analysis import stemminganalyzer import os define schema for indexing schema |
|           | schema text stored true content text stored true analyzer stemminganalyzer stemming for better       |
|           | search results path id stored true unique true create index directory if it doesn t exist if not os  |
|           | path exists indexdir os mkdi...                                                                      |
| embedding | [0.0024, -0.0019, -0.0199, 0.0600, -0.0127, -0.0041, 0.0037, -0.0299, -0.0054, 0.0241, 0.0551,       |
|           | 0.0135, 0.0307, -0.0229, -0.0594, 0.0180, -0.0604, -0.0051, -0.0513, -0.0996, 0.0903, 0.0993,        |
|           | 0.0307, -0.0362, -0.0801, 0.0011, -0.0448, 0.0053, -0.0028, -0.0308, 0.0065, 0.1206, 0.0187, 0.0684, |
|           | -0.0024, 0.0298, 0.0412, -0.0044, -0.0011, -0.0056, 0.0066, -0.0152, -0.0298, 0.0541, -0.0385,       |
|           | -0.0145, -0.0695, -0.0408, 0.0137, -0.0141, -0.1621, 0.0138, -0.0047, 0.0533, 0.0743, 0.0738,        |
|           | -0.0567, 0.0371, -0.0554, -0.0892, 0.0054, 0.0133, -0.0204, -0.0234... (384 total)]                  |
+-----------+------------------------------------------------------------------------------------------------------+
```

