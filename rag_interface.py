#!/usr/bin/env python3

import os
import subprocess
import sys

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("=" * 50)
    print("RAG Search System")
    print("=" * 50)
    print("Options:")
    print("1. CLI Search Interface")
    print("2. Start API Server")
    print("3. Exit")
    print("=" * 50)

def main():
    while True:
        clear_screen()
        print_header()
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            subprocess.run([sys.executable, "cli_search.py"])
        elif choice == '2':
            try:
                subprocess.run([sys.executable, "api_server.py"])
            except KeyboardInterrupt:
                print("\nStopping API server...")
        elif choice == '3':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
