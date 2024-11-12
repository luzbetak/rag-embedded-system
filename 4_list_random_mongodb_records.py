#!/usr/bin/env python3

from pymongo import MongoClient
from prettytable import PrettyTable
import numpy as np
from core.config import Config

def format_embedding(embedding, num_values=64):
    """Format embedding vector to show first num_values with better precision"""
    if isinstance(embedding, (list, np.ndarray)):
        values = [f"{x:.4f}" for x in embedding[:num_values]]
        return f"[{', '.join(values)}" + f"... ({len(embedding)} total)]"
    return str(embedding)

def format_value(value, max_length=700):
    """Format values for display, truncating if necessary"""
    if isinstance(value, (list, np.ndarray)):
        return format_embedding(value)

    str_value = str(value)
    if len(str_value) > max_length:
        return str_value[:max_length] + "..."
    return str_value

def explore_mongodb():
    """
    Display MongoDB records with specific columns and formatting
    """
    try:
        client = MongoClient(Config.MONGODB_URI)
        print("\n✅ Successfully connected to MongoDB\n")
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return

    # Use configured database and collection
    db = client[Config.DATABASE_NAME]
    collection = db[Config.COLLECTION_NAME]

    # Get three random samples
    samples = list(collection.aggregate([{"$sample": {"size": 3}}]))

    if not samples:
        print("No documents found in collection")
        return

    # Create main table for the records
    main_table = PrettyTable()
    main_table.field_names = ["Field", "Value"]
    main_table.align = 'l'  # Left align text

    # Set individual column widths
    main_table.max_width["Field"] = 15
    main_table.max_width["Value"] = 160 

    # Display each sample
    for i, sample in enumerate(samples, 1):
        # print(f"\n🎲 Sample Record #{i}:")
        main_table.clear_rows()

        # Add rows for specified fields
        fields = ['_id', 'url', 'title', 'content', 'embedding']
        for field in fields:
            if field in sample:
                value = format_value(sample[field])
                main_table.add_row([field, value])

        print(main_table)

if __name__ == "__main__":
    explore_mongodb()

