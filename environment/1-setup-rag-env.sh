#!/bin/bash

# Colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to create environment
create_env() {
    echo -e "${BLUE}Creating new RAG environment with Python 3.11...${NC}"
    conda create -n rag python=3.11 -y
    echo -e "${GREEN}Environment created successfully!${NC}"
}

# Function to activate environment
activate_env() {
    echo -e "${BLUE}Activating RAG environment...${NC}"
    conda activate rag
    echo -e "${GREEN}Environment activated!${NC}"
}

# Function to install dependencies
install_deps() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    echo "Installing core dependencies..."
    conda install -c conda-forge pandas sqlalchemy sentence-transformers fastapi uvicorn python-dotenv loguru numpy pymongo spacy nltk -y
    
    echo "Installing development packages..."
    conda install -c conda-forge jupyter ipython black pytest sumy rouge fastapi loguru matplotlib -y
    
    echo -e "${BLUE}Verifying Python version...${NC}"
    python --version
    
    echo -e "${BLUE}Verifying installations...${NC}"
    python -c "import pandas, sqlalchemy, sentence_transformers, fastapi, uvicorn, numpy; print('All packages installed successfully!')"
    
    echo -e "${GREEN}All dependencies installed successfully!${NC}"
}

# Function to install MongoDB
install_mongodb() {
    echo -e "${BLUE}Installing MongoDB...${NC}"
    
    # Check if MongoDB is already installed
    if systemctl is-active --quiet mongod; then
        echo -e "${GREEN}MongoDB is already installed and running!${NC}"
        return 0
    fi
    
    # Import MongoDB public GPG key
    echo "Importing MongoDB public GPG key..."
    curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
        sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
        --dearmor
    
    # Create list file for MongoDB
    echo "Creating MongoDB list file..."
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | \
        sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    
    # Update package database
    echo "Updating package database..."
    sudo apt-get update
    
    # Install MongoDB
    echo "Installing MongoDB packages..."
    sudo apt-get install -y mongodb-org
    
    # Start MongoDB
    echo "Starting MongoDB service..."
    sudo systemctl start mongod
    
    # Enable MongoDB to start on boot
    echo "Enabling MongoDB on boot..."
    sudo systemctl enable mongod
    
    # Verify MongoDB is running
    echo "Verifying MongoDB status..."
    if systemctl is-active --quiet mongod; then
        echo -e "${GREEN}MongoDB installed and running successfully!${NC}"
        sudo systemctl status mongod
    else
        echo -e "${RED}MongoDB installation might have failed. Please check the logs.${NC}"
        return 1
    fi
}

# Function to remove environment
remove_env() {
    echo -e "${RED}Deactivating and removing RAG environment...${NC}"
    conda deactivate
    conda env remove -n rag -y
    echo -e "${GREEN}Environment removed successfully!${NC}"
}

# Function to remove MongoDB
remove_mongodb() {
    echo -e "${RED}Removing MongoDB...${NC}"
    
    # Stop MongoDB service
    echo "Stopping MongoDB service..."
    sudo systemctl stop mongod
    
    # Disable MongoDB from starting on boot
    echo "Disabling MongoDB service..."
    sudo systemctl disable mongod
    
    # Remove MongoDB packages
    echo "Removing MongoDB packages..."
    sudo apt-get purge -y mongodb-org*
    
    # Remove MongoDB data and log files
    echo "Removing MongoDB data and log files..."
    sudo rm -rf /var/log/mongodb
    sudo rm -rf /var/lib/mongodb
    
    echo -e "${GREEN}MongoDB removed successfully!${NC}"
}

# Main menu
while true; do
    echo -e "\n${BLUE}RAG Environment Management${NC}"
    echo "1. Create new RAG environment"
    echo "2. Activate RAG environment"
    echo "3. Install dependencies"
    echo "4. Install MongoDB"
    echo "5. Remove RAG environment"
    echo "6. Remove MongoDB"
    echo "7. Exit"
    
    read -p "Please select an option (1-7): " choice
    
    case $choice in
        1)
            create_env
            ;;
        2)
            activate_env
            ;;
        3)
            install_deps
            ;;
        4)
            install_mongodb
            ;;
        5)
            remove_env
            ;;
        6)
            remove_mongodb
            ;;
        7)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please choose 1-7${NC}"
            ;;
    esac
done

