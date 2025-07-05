from pydantic import BaseModel
from typing import List, Optional, Dict, Literal, Any
from datetime import datetime

class QuestionRequest(BaseModel):
    question: str
    city: Optional[str] = None  # If specified, filter by city
    category: Optional[str] = None  # If specified, filter by category/section
    context_limit: Optional[int] = 5  # Number of context chunks to retrieve
    search_type: Optional[Literal["vector", "keyword", "hybrid"]] = "hybrid"  # Search type
    use_reranking: Optional[bool] = True  # Whether to use Cohere reranking
    stream: Optional[bool] = False  # Whether to stream the response

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None
    
    # Enhanced jurisdiction fields
    jurisdiction_level: Optional[str] = None  # "district", "city", "county", "state"
    jurisdiction_name: Optional[str] = None   # "Downtown District", "Sunnyvale", "California"
    jurisdiction_priority: Optional[int] = None  # 1=highest (district), 4=lowest (state)
    regulation_type: Optional[str] = None     # "zoning", "parking", "setbacks", etc.
    
    # Conflict resolution fields
    numeric_requirements: Optional[Dict[str, float]] = None  # {"min_setback": 10, "max_height": 35}
    conflicts_with: Optional[List[str]] = None  # List of conflicting jurisdiction IDs
    controlling_authority: Optional[bool] = None  # True if this regulation controls in conflicts
    
    # Enhanced citation info
    title: Optional[str] = None
    city: Optional[str] = None
    category: Optional[str] = None
    chunk_index: Optional[int] = None
    source_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str
    city_filter: Optional[str] = None
    category_filter: Optional[str] = None
    search_type: str = "hybrid"
    reranking_used: bool = False
    timestamp: datetime

class StreamingChunk(BaseModel):
    chunk: str
    is_final: bool = False
    sources: Optional[List[SourceDocument]] = None

class DocumentUpload(BaseModel):
    title: str
    content: str
    city: str
    category: str  # e.g., "zoning", "parking", "street_standards"
    metadata: Optional[Dict] = None
    source_url: Optional[str] = None  # Optional URL for citations

class DocumentInfo(BaseModel):
    id: str
    title: str
    city: str
    category: str
    created_at: datetime
    metadata: Dict

class HybridSearchRequest(BaseModel):
    query: str
    city: Optional[str] = None
    category: Optional[str] = None
    limit: Optional[int] = 10
    alpha: Optional[float] = 0.7  # Weight for vector vs keyword search (0.7 = 70% vector, 30% keyword) 