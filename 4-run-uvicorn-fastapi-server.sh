#!/usr/bin/env python3

import uvicorn
from query import app
import webbrowser
import threading
import time

# uvicorn query:app --reload

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:8000/docs')

def main():
    print("\nStarting RAG API Server...")
    print("You can access the API documentation at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    # Open browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

