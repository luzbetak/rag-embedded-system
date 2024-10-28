# RAG System

A Retrieval-Augmented Generation (RAG) system using MongoDB for document storage and sentence-transformers for embeddings.


# 3-init-rag-db.py

## RAG System Database Initialization Guide

`3-init_rag_db.py` is the database initialization component for the RAG (Retrieval-Augmented Generation) system. It handles setting up MongoDB collections, loading documents, and generating embeddings for semantic search.

## Database Configuration
The system uses MongoDB with the following default settings:
```python
MONGODB_URI = "mongodb://localhost:27017"
DATABASE_NAME = "rag_database"
COLLECTION_NAME = "documents"
```

## Prerequisites
1. MongoDB installed and running
2. Python 3.7+
3. Required Python packages:
   ```bash
   pip install pymongo sentence-transformers loguru numpy python-dotenv
   ```

## File Structure
```
project/
├── 3-init_rag_db.py          # Main initialization script
├── config.py               # Configuration settings
├── database.py            # MongoDB connection handling
├── data_ingestion.py      # Document processing pipeline
├── vectorization.py       # Embedding generation
└── data/
    └── search-index.json  # Source documents
```

## Document Format
Your `search-index.json` should follow this structure:
```json
[
  {
    "title": "Document Title",
    "content": "Document Content",
    "metadata": {
      "key1": "value1",
      "key2": "value2"
    }
  }
]
```

## Script Components

### 1. Database Initialization
```python
def init_database():
    """
    Initializes fresh MongoDB collection with required indices
    - Drops existing collection if present
    - Creates indices for 'title' and 'content'
    """
```

### 2. Document Loading
```python
def load_documents():
    """
    Processes documents and generates embeddings
    - Loads documents from search-index.json
    - Preprocesses document text
    - Generates embeddings using sentence-transformers
    - Stores documents with embeddings in MongoDB
    """
```

## Usage

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start MongoDB:
   ```bash
   # Linux/MacOS
   sudo service mongodb start
   
   # Windows
   net start MongoDB
   ```

### Running the Script
```bash
python 3-init_rag_db.py
```

The script provides three options:
1. Initialize database (deletes existing data)
2. Load documents from search-index.json
3. Exit

### First Time Setup
```bash
python 3-init_rag_db.py
# Choose option 1 to initialize database
# Choose option 2 to load documents
```

### Updating Documents
```bash
# Choose option 2 to load new/updated documents
```

## Environment Variables
The script uses these environment variables to optimize performance:
```bash
OPENBLAS_NUM_THREADS=1
OPENBLAS_MAIN_FREE=1
OMP_NUM_THREADS=1
```


### MongoDB Installation

#### Ubuntu
```bash
sudo apt-get update
sudo apt-get install mongodb
sudo systemctl start mongodb
```

#### MacOS
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

#### Windows
1. Download MongoDB Community Server from MongoDB website
2. Run the installer
3. Add MongoDB to system PATH
4. Create data directory: `C:\data\db`

## Additional Notes

- The script uses sentence-transformers for generating embeddings
- Default model: 'sentence-transformers/all-MiniLM-L6-v2'
- Embedding dimension: 384
- MongoDB indices optimize search performance
- Logging is handled by loguru for better debugging

## Support
For additional help or issues:
1. Check MongoDB logs: `/var/log/mongodb/mongodb.log`
2. Check RAG system logs: `pipeline.log`
3. Ensure all dependencies are properly installed
4. Verify MongoDB service is running

------------------------------------------------------------------------------------------------------------------
