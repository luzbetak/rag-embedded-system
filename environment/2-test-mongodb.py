#!/usr/bin/env python

from pymongo import MongoClient

# Create a client connection
client = MongoClient('mongodb://localhost:27017/')

# Test the connection
try:
    client.admin.command('ping')
    print("MongoDB connection successful!")
except Exception as e:
    print(f"MongoDB connection failed: {e}")
