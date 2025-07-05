from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
import logging
import json

from ...models.schemas import (
    QuestionRequest, 
    ChatResponse, 
    SourceDocument, 
    DocumentUpload,
    DocumentInfo,
    StreamingChunk,
    HybridSearchRequest
)
from ...services.rag_service import EnhancedRAGService

logger = logging.getLogger(__name__)
router = APIRouter()

# Global RAG service instance
rag_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        rag_service = EnhancedRAGService()
    return rag_service

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about Bay Area city regulations"""
    try:
        rag = get_rag_service()
        
        # If streaming is requested, return streaming response
        if request.stream:
            return await ask_question_streaming(request)
        
        response = rag.query(request)
        return response
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@router.post("/ask/stream")
async def ask_question_streaming(request: QuestionRequest):
    """Ask a question with streaming response"""
    try:
        rag = get_rag_service()
        
        async def generate_stream():
            async for chunk in rag.query_streaming(request):
                # Format as Server-Sent Events
                chunk_data = {
                    "chunk": chunk.chunk,
                    "is_final": chunk.is_final,
                    "sources": [source.dict() for source in (chunk.sources or [])]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                
                if chunk.is_final:
                    break
        
        return StreamingResponse(
            generate_stream(), 
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in streaming: {str(e)}")

@router.post("/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """Perform hybrid search combining vector and keyword search"""
    try:
        rag = get_rag_service()
        documents = rag.hybrid_search(request)
        return {
            "documents": documents, 
            "query": request.query, 
            "search_type": "hybrid",
            "city_filter": request.city,
            "category_filter": request.category,
            "alpha": request.alpha
        }
        
    except Exception as e:
        logger.error(f"Error in hybrid search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in hybrid search: {str(e)}")

@router.post("/documents")
async def upload_documents(documents: List[DocumentUpload], background_tasks: BackgroundTasks):
    """Upload regulation documents to the vector database"""
    try:
        rag = get_rag_service()
        
        # Convert to the format expected by RAGService
        doc_dicts = []
        for doc in documents:
            doc_dicts.append({
                "title": doc.title,
                "content": doc.content,
                "city": doc.city,
                "category": doc.category,
                "source_url": doc.source_url,
                "metadata": doc.metadata or {}
            })
        
        # Add documents in background
        background_tasks.add_task(rag.add_documents, doc_dicts)
        
        return {"message": f"Queued {len(documents)} documents for processing"}
        
    except Exception as e:
        logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")

@router.get("/search")
async def search_documents(
    query: str, 
    city: Optional[str] = None, 
    category: Optional[str] = None,
    limit: int = 5,
    search_type: str = "vector"
):
    """Search for similar documents without generating an answer"""
    try:
        rag = get_rag_service()
        
        if search_type == "hybrid":
            # Use hybrid search
            request = HybridSearchRequest(
                query=query,
                city=city,
                category=category,
                limit=limit
            )
            documents = rag.hybrid_search(request)
        else:
            # Use vector search
            documents = rag.get_similar_documents(
                query, k=limit, city=city, category=category
            )
        
        return {
            "documents": documents, 
            "query": query, 
            "city_filter": city,
            "category_filter": category,
            "search_type": search_type
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@router.get("/stats")
async def get_database_stats():
    """Get statistics about the regulation database"""
    try:
        rag = get_rag_service()
        stats = rag.get_database_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        rag = get_rag_service()
        stats = rag.get_database_stats()
        return {
            "status": "healthy",
            "database_connected": True,
            "total_documents": stats.get("total_documents", 0),
            "hybrid_search_enabled": stats.get("hybrid_search_enabled", False),
            "reranking_enabled": stats.get("reranking_enabled", False),
            "features": {
                "vector_search": True,
                "keyword_search": stats.get("hybrid_search_enabled", False),
                "hybrid_search": stats.get("hybrid_search_enabled", False),
                "reranking": stats.get("reranking_enabled", False),
                "streaming": True,
                "category_filtering": True,
                "hierarchy": True
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database_connected": False,
            "error": str(e)
        }

@router.post("/ask/hierarchy")
async def ask_question_with_hierarchy(request: QuestionRequest):
    """
    Ask a question with regulatory hierarchy analysis
    
    This endpoint provides enhanced responses that consider regulatory precedence:
    - Special Districts/Master Plans (highest authority)
    - City/Municipal regulations  
    - County regulations
    - State regulations (lowest authority)
    """
    try:
        result = get_rag_service().query_with_hierarchy(request)
        return result
        
    except Exception as e:
        logger.error(f"Error processing hierarchical question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 