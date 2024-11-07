#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from prettytable import PrettyTable
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Enhanced input texts with related concepts
texts = [
    "cat",      "kitten",     "dog",        "puppy",     # Animals
    "computer", "laptop",     "smartphone", "macbook",   # Technology
    "apple",    "banana",     "fruit",      "pineapple", # Food
    "car",      "automobile", "vehicle",    "airplane",  # Transportation
]

# Get embeddings
embeddings = model.encode(texts)

main_table  = PrettyTable()
field_names = ["Text", "Length"]
field_names.extend([f"Dim_{i+1}" for i in range(12)])
main_table.field_names = field_names
main_table.align["Text"]   = "r"
main_table.align["Length"] = "c"

for i in range(12):
    main_table.align[f"Dim_{i+1}"] = "c"

# Add rows to main table
for text, embedding in zip(texts, embeddings):
    row_data = [text, len(embedding)]
    row_data.extend([f"{num:.3f}" for num in embedding[:12]])
    main_table.add_row(row_data)

# Calculate similarity matrix
similarity_matrix = np.zeros((len(texts), len(texts)))
for i in range(len(texts)):
    for j in range(len(texts)):
        similarity_matrix[i][j] = 1 - distance.cosine(embeddings[i], embeddings[j])

# Create similarity heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', 
            xticklabels=texts, yticklabels=texts, fmt='.2f')
plt.title('Word Similarity Heatmap')
plt.tight_layout()
plt.savefig('similarity_heatmap.png')
plt.close()

# Perform PCA for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(StandardScaler().fit_transform(embeddings))

# Create scatter plot of words in 2D space
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
for i, text in enumerate(texts):
    plt.annotate(text, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.title('Word Embeddings Visualized in 2D')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.tight_layout()
plt.savefig('embeddings_2d.png')
plt.close()

# Print results
print("\nText Embedding Analysis (First 12 Dimensions):")
print(main_table)

# Create enhanced similarity analysis
print("\nDetailed Similarity Analysis:")
print("-" * 50)

# Group similar concepts
concept_groups = [
    ("cat",      "kitten"),
    ("dog",      "puppy"),
    ("computer", "laptop",     "macbook"),
    ("car",      "automobile", "vehicle"),
    ("apple",    "banana",     "fruit")
]

for group in concept_groups:
    group_table = PrettyTable()
    group_table.field_names = ["Word Pair", "Similarity", "Avg Magnitude", "Dimension Correlation"]
    
    # Calculate group statistics
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            idx1 = texts.index(group[i])
            idx2 = texts.index(group[j])
            
            similarity      = 1 - distance.cosine(embeddings[idx1], embeddings[idx2])
            avg_magnitude   = (np.linalg.norm(embeddings[idx1]) + np.linalg.norm(embeddings[idx2])) / 2
            dim_correlation = np.corrcoef(embeddings[idx1], embeddings[idx2])[0, 1]
            
            group_table.add_row([
                f"{group[i]} - {group[j]}",
                f"{similarity:.3f}",
                f"{avg_magnitude:.3f}",
                f"{dim_correlation:.3f}"
            ])
    
    print(f"\nAnalysis for concept group: {', '.join(group)}")
    print(group_table)

# Additional statistical measures
print("\nAdvanced Statistical Analysis:")
print("-" * 50)

stats_table = PrettyTable()
stats_table.field_names = ["Text", "Magnitude", "Mean", "Std Dev", "Max Dim", "Min Dim"]
stats_table.align = "r"

for text, embedding in zip(texts, embeddings):
    magnitude = np.linalg.norm(embedding)
    mean_val  = np.mean(embedding)
    std_dev   = np.std(embedding)
    max_dim   = np.max(embedding)
    min_dim   = np.min(embedding)
    
    stats_table.add_row([
        text,
        f"{magnitude:.3f}",
        f"{mean_val:.3f}",
        f"{std_dev:.3f}",
        f"{max_dim:.3f}",
        f"{min_dim:.3f}"
    ])

print("\nStatistical Measures per Word:")
print(stats_table)

# Dimension importance analysis
print("\nDimension Importance Analysis:")
print("-" * 50)

# Calculate variance explained by each dimension
variance_explained = np.var(embeddings, axis=0)
top_dims = np.argsort(variance_explained)[-10:][::-1]

var_table = PrettyTable()
var_table.field_names = ["Dimension", "Variance", "% of Total Variance"]
var_table.align = "r"

total_variance = np.sum(variance_explained)
for dim in top_dims:
    var_table.add_row([
        f"Dim_{dim+1}",
        f"{variance_explained[dim]:.3f}",
        f"{(variance_explained[dim]/total_variance)*100:.2f}%"
    ])

print("\nTop 10 Most Important Dimensions:")
print(var_table)

print("\nVisualization files generated:")
print("1. similarity_heatmap.png - Shows similarity between all word pairs")
print("2. embeddings_2d.png - Shows words plotted in 2D space using PCA")

