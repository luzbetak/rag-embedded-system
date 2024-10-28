# query.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from typing import List, Optional
from database import Database
from vectorization import VectorizationPipeline
from transformers import pipeline

# Set OpenBLAS environment variables
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OPENBLAS_MAIN_FREE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Global QueryEngine instance
query_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for the FastAPI application"""
    global query_engine
    
    # Startup
    logger.info("Initializing RAG Search API...")
    query_engine = QueryEngine()
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Search API...")
    if query_engine:
        query_engine.close()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Search API",
    description="API for vector similarity search and response generation",
    version="1.0.0",
    lifespan=lifespan
)

# Rest of your QueryEngine class and endpoint definitions remain the same
class SearchRequest(BaseModel):
    text: str
    top_k: Optional[int] = 3

class SearchResponse(BaseModel):
    similar_documents: List[dict]
    generated_response: str

class HealthCheckResponse(BaseModel):
    status: str
    version: str

class QueryEngine:
    def __init__(self):
        """Initialize the query engine with necessary components"""
        self.db = Database()
        self.vectorization = VectorizationPipeline()
        self.generator = pipeline('text2text-generation', model='google/flan-t5-base', max_length=200)
        logger.info("Initialized query engine")

    async def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Perform vector similarity search"""
        try:
            # Generate query embedding
            query_embedding = self.vectorization.generate_embeddings([query])[0]
            
            # Get similar documents
            similar_docs = self.db.get_similar_documents(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    async def generate_response(self, query: str, documents: List[dict]) -> str:
        """Generate a response based on the query and retrieved documents"""
        try:
            # Combine document contents
            context = "\n".join([
                f"Document {i+1}: {doc.get('content', '')}"
                for i, doc in enumerate(documents)
            ])
            
            # Create prompt for the generator
            prompt = (
                f"Based on the following context, answer the question: {query}\n\n"
                f"Context:\n{context}\n\n"
                f"Answer:"
            )
            
            # Generate response
            response = self.generator(prompt)[0]['generated_text']
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return "I apologize, but I encountered an error generating a response."

    def close(self):
        """Cleanup resources"""
        self.db.close()

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint that performs vector similarity search and generates a response"""
    try:
        # Perform vector similarity search
        similar_docs = await query_engine.search(request.text, top_k=request.top_k)
        
        if not similar_docs:
            return SearchResponse(
                similar_documents=[],
                generated_response="No relevant documents found for your query."
            )
        
        # Generate a response based on similar documents
        generated_response = await query_engine.generate_response(
            query=request.text,
            documents=similar_docs
        )
        
        return SearchResponse(
            similar_documents=similar_docs,
            generated_response=generated_response
        )
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )

