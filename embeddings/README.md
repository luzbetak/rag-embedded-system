Word Embedding Analysis Tool
============================

A Python script for analyzing and visualizing semantic relationships between words using sentence transformers. This tool generates word embeddings, calculates similarities between related concepts, and provides various visualizations and statistical analyses.

## Features

- Generates word embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model
- Calculates word similarities using cosine similarity (1 - cosine_distance)
- Creates detailed similarity matrices and statistical analyses
- Visualizes relationships between words using heatmaps and 2D scatter plots
- Groups and analyzes related concepts
- Provides comprehensive statistical measures for each word embedding

## Similarity Calculation

The similarity between words is computed using cosine similarity, which measures the cosine of the angle between two word embedding vectors. The formula used is:

```
similarity = 1 - cosine_distance
```

This results in:
- A value of 1 for identical word embeddings
- Values close to 1 for very similar words
- Values close to 0 for unrelated words
- Values close to -1 for opposite meanings

## Example Output

Below is a similarity heatmap showing the semantic relationships between different words:

![Word Similarity Heatmap](similarity_heatmap.png)

The heatmap visualizes the cosine similarity between pairs of words, where darker red indicates higher similarity and darker blue indicates lower similarity. Related concepts like "car"/"automobile" and "cat"/"kitten" show strong semantic connections.

## Dependencies

- sentence-transformers
- numpy
- prettytable
- scipy
- matplotlib
- seaborn
- scikit-learn

## Output Files

The script generates two visualization files:
1. `similarity_heatmap.png` - Heatmap showing similarity between all word pairs
2. `embeddings_2d.png` - 2D scatter plot of words using PCA dimensionality reduction
