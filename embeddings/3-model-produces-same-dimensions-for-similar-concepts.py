#!/usr/bin/env python

from sentence_transformers import SentenceTransformer
import numpy as np
from prettytable import PrettyTable
from scipy.spatial import distance

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Input texts
texts = [
    "cat",
    "kitten",
    "computer",
]

# Get embeddings
embeddings = model.encode(texts)

# Create main table for embeddings
main_table = PrettyTable()

# Define column names
field_names = ["Text", "Length"]
field_names.extend([f"Dim_{i+1}" for i in range(12)])  # Show first 12 dimensions
main_table.field_names = field_names

# Set alignment
main_table.align["Text"] = "r"
main_table.align["Length"] = "c"
for i in range(12):
    main_table.align[f"Dim_{i+1}"] = "c"

# Add rows to main table
for text, embedding in zip(texts, embeddings):
    row_data = [text, len(embedding)]
    row_data.extend([f"{num:.3f}" for num in embedding[:12]])
    main_table.add_row(row_data)

# Create similarity table
sim_table = PrettyTable()
sim_table.field_names = ["Text 1", "Text 2", "Cosine Similarity"]
sim_table.align = "r"

# Calculate similarities between all pairs
for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = 1 - distance.cosine(embeddings[i], embeddings[j])  # Corrected cosine distance calculation
        sim_table.add_row([
            texts[i],
            texts[j],
            f"{similarity:.3f}"
        ])

# Print results
print("\nText Embedding Analysis (First 12 Dimensions):")
print(main_table)

print("\nPairwise Cosine Similarities:")
print(sim_table)

# Additional statistical analysis
print("\nStatistical Analysis:")
print("-" * 50)

# Calculate vector magnitudes
magnitudes = np.linalg.norm(embeddings, axis=1)
print("Vector Magnitudes:")
mag_table = PrettyTable()
mag_table.field_names = ["Text", "Magnitude"]
mag_table.align = "r"
for text, magnitude in zip(texts, magnitudes):
    mag_table.add_row([text, f"{magnitude:.3f}"])
print(mag_table)

# Find dimensions with highest variance
variances = np.var(embeddings, axis=0)
top_variance_dims = np.argsort(variances)[-5:][::-1]
print("\nTop 5 Dimensions by Variance:")
var_table = PrettyTable()
var_table.field_names = ["Dimension", "Variance"]
var_table.align = "r"
for dim in top_variance_dims:
    var_table.add_row([f"Dim_{dim+1}", f"{variances[dim]:.3f}"])
print(var_table)

