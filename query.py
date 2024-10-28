# query.py
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from config import Config
from vectorization import VectorizationPipeline
from database import Database

app = FastAPI()

class Query(BaseModel):
    text: str
    top_k: Optional[int] = Config.TOP_K

class QueryEngine:
    def __init__(self):
        self.vectorization = VectorizationPipeline()
        self.db = Database()
        logger.info("Initialized query engine")
    
    async def search(self, query_text: str, top_k: int = Config.TOP_K) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity"""
        query_embedding = self.vectorization.model.encode([query_text])[0].tolist()
        similar_docs = self.db.get_similar_documents(query_embedding, top_k)
        return similar_docs
    
    async def generate_response(self, query_text: str, similar_docs: List[Dict[str, Any]]) -> str:
        """Generate a response using retrieved documents (RAG)"""
        context = "\n".join([doc["content"] for doc in similar_docs])
        response = f"Based on {len(similar_docs)} relevant documents, here's a summary..."
        return response

query_engine = QueryEngine()

@app.post("/search")
async def search_documents(query: Query):
    try:
        similar_docs = await query_engine.search(query.text, query.top_k)
        response = await query_engine.generate_response(query.text, similar_docs)
        return {
            "similar_documents": similar_docs,
            "generated_response": response
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

