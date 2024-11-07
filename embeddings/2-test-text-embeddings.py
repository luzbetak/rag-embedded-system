#!/usr/bin/env python

from sentence_transformers import SentenceTransformer
import numpy as np
from prettytable import PrettyTable

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Different types of text input
texts = [
    "My Cat",                                   # Single word
    "The cat sits on the mat",                  # Short sentence
    "Cats love to play and sleep all day long." # Longer paragraph
]

# Create PrettyTable
table = PrettyTable()

# Define column names
field_names = ["Text", "Length"]
field_names.extend([f"Dim_{i+1}" for i in range(12)])  # Add columns for each dimension

table.field_names = field_names

# Set alignment for columns
table.align["Text"]   = "l"  # Left align text
table.align["Length"] = "c"  # Center align length

for i in range(12):
    table.align[f"Dim_{i+1}"] = "c"  # Center align all dimension columns

# Convert each text to embeddings
for text in texts:
    embedding = model.encode(text)
    
    # Prepare row data
    row_data = [
        text[:40] + "..." if len(text) > 40 else text,  # Truncate long text
        len(embedding)
    ]
    
    # Add first 12 embedding numbers
    row_data.extend([f"{num:.3f}" for num in embedding[:12]])
    
    # Add row to table
    table.add_row(row_data)

# Print the table
print("\nText Embedding Analysis:")
print(table)

# Additional Statistics
print("\nEmbedding Statistics:")
all_embeddings = model.encode(texts)
print(f"Shape of all embeddings: {all_embeddings.shape}")
print(f"Average vector magnitude: {np.mean([np.linalg.norm(emb) for emb in all_embeddings]):.3f}")

