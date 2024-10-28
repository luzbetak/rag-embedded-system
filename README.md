# Embedded RAG System

A Retrieval-Augmented Generation (RAG) system using MongoDB for document storage and sentence-transformers for embeddings.

## Prerequisites

- Ubuntu/Linux system
- Anaconda/Miniconda
- Python 3.11
- MongoDB

## Installation Steps

### 1. MongoDB Installation

```bash
# Import MongoDB public GPG key
curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
   --dearmor

# Create list file for MongoDB
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Reload local package database
sudo apt-get update

# Install MongoDB packages
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod

# Enable MongoDB to start on boot
sudo systemctl enable mongod

# Verify MongoDB is running
sudo systemctl status mongod
```

### 2. Create Conda Environment

```bash
# Create new environment
conda create -n rag python=3.11

# Activate environment
conda activate rag

# Install required packages
conda install -c conda-forge pandas sqlalchemy sentence-transformers fastapi uvicorn python-dotenv loguru numpy pymongo jupyter
```

### 3. Project Setup

```bash
# Clone or create project directory
mkdir embedded-rag-system
cd embedded-rag-system

# Create required directories
mkdir -p data logs
```

### 4. Initialize MongoDB Database and Collection

```python
# Run Python in your rag environment
python

# In Python console:
from database import Database
db = Database()
db.collection.drop()  # If you need to reset the database
```

### 5. Prepare Test Documents

Create a file `data/documents.json` with some test documents:

```json
[
    {
        "title": "Test Document 1",
        "content": "This is a test document for the RAG system.",
        "metadata": {
            "author": "Test Author",
            "date": "2024-10-27"
        }
    },
    {
        "title": "Test Document 2",
        "content": "Another test document to verify the pipeline.",
        "metadata": {
            "author": "Test Author 2",
            "date": "2024-10-27"
        }
    }
]
```

### 6. File Structure

Ensure you have all the required Python files:

```
embedded-rag-system/
├── config.py           # Configuration settings
├── database.py        # Database operations
├── data_ingestion.py  # Data loading and preprocessing
├── vectorization.py   # Embedding generation
├── query.py           # Search and retrieval
├── main.py           # Main pipeline script
├── search.py         # Search interface
├── data/             # Document storage
│   └── documents.json
└── logs/             # Log files
    └── pipeline.log
```

### 7. Run the Pipeline

```bash
# Make sure you're in the rag environment
conda activate rag

# Run the main pipeline to process documents
python main.py
```

### 8. Search Documents

```bash
# Run the search interface
python search.py
```

## Usage

1. To add new documents:
   - Add them to `data/documents.json`
   - Run `main.py` to process them

2. To search documents:
   - Run `search.py`
   - Choose option 1 to search
   - Enter your query
   - View results

3. To reset the system:
   ```python
   from database import Database
   db = Database()
   db.collection.drop()
   ```
   Then run `main.py` again to reload documents.

## Troubleshooting

1. If MongoDB isn't running:
   ```bash
   sudo systemctl start mongod
   sudo systemctl status mongod
   ```

2. To check MongoDB logs:
   ```bash
   sudo tail -f /var/log/mongodb/mongod.log
   ```

3. To verify database contents:
   ```python
   from database import Database
   db = Database()
   list(db.collection.find({}, {'embedding': 0}))  # View documents without embeddings
   ```

## Project Components

- `config.py`: Configuration settings for MongoDB and model parameters
- `database.py`: MongoDB database operations and similarity search
- `data_ingestion.py`: Document loading and preprocessing
- `vectorization.py`: Document embedding generation
- `query.py`: Search functionality and API
- `main.py`: Main pipeline execution
- `search.py`: Interactive search interface

## Notes

- The system uses sentence-transformers for generating embeddings
- Documents are stored in MongoDB with their embeddings
- Similarity search is performed using cosine similarity
- Duplicate documents are automatically handled
- The system maintains logs in the `logs` directory

## Future Improvements

- Add support for more document formats
- Implement better text preprocessing
- Add filtering options in search
- Add batch processing for large document sets
- Implement document updating functionality



Would you like me to:
1. Add more details to any section?
2. Add configuration examples?
3. Add more troubleshooting steps?
4. Add example code for common tasks?# embedded-rag-system
