#!/usr/bin/env python

from sentence_transformers import SentenceTransformer

# This will download the model to your local cache (usually ~/.cache/torch/sentence_transformers/)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Test the model with a simple example
sentences = ['This is a test sentence.', 'Another test sentence to encode.']
embeddings = model.encode(sentences)

print(f"Model loaded successfully!")
print(f"Embedding shape: {embeddings.shape}")  # Should show (2, 384)
