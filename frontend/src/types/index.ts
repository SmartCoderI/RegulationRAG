export interface QuestionRequest {
  question: string;
  city?: string;
  category?: string;
  context_limit?: number;
  search_type?: "vector" | "keyword" | "hybrid";
  use_reranking?: boolean;
  stream?: boolean;
}

export interface SourceDocument {
  content: string;
  metadata: Record<string, any>;
  score?: number;
  title?: string;
  city?: string;
  category?: string;
  chunk_index?: number;
  source_url?: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceDocument[];
  query: string;
  city_filter?: string;
  category_filter?: string;
  search_type: string;
  reranking_used: boolean;
  timestamp: string;
}

export interface StreamingChunk {
  chunk: string;
  is_final: boolean;
  sources?: SourceDocument[];
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: SourceDocument[];
  city_filter?: string;
  category_filter?: string;
  search_type?: string;
  reranking_used?: boolean;
  is_streaming?: boolean;
}

export interface DatabaseStats {
  total_documents: number;
  total_chunks?: number;
  cities: string[];
  categories: string[];
  hybrid_search_enabled?: boolean;
  reranking_enabled?: boolean;
  features?: {
    vector_search: boolean;
    keyword_search: boolean;
    hybrid_search: boolean;
    reranking: boolean;
    streaming: boolean;
    category_filtering: boolean;
    hierarchy: boolean;
  };
}

export interface DocumentUpload {
  title: string;
  content: string;
  city: string;
  category: string;
  source_url?: string;
  metadata?: Record<string, any>;
}

export interface HybridSearchRequest {
  query: string;
  city?: string;
  category?: string;
  limit?: number;
  alpha?: number;
}

export interface ApiError {
  detail: string;
}

// Enhanced response for hierarchical queries
export interface HierarchicalResponse {
  answer: string;
  sources: SourceDocument[];
  hierarchy_explanation: string;
  controlling_regulations: SourceDocument[];
  supporting_regulations: SourceDocument[];
  conflicts_detected: boolean;
  query: string;
  city_filter?: string;
  category_filter?: string;
  search_type: string;
  reranking_used: boolean;
  timestamp: string;
}

// Enhanced source document with jurisdiction info
export interface EnhancedSourceDocument extends SourceDocument {
  jurisdiction_level?: string;
  jurisdiction_name?: string;
  jurisdiction_priority?: number;
  regulation_type?: string;
  numeric_requirements?: Record<string, number>;
  conflicts_with?: string[];
  controlling_authority?: boolean;
} 