import axios from 'axios';
import { 
  QuestionRequest, 
  ChatResponse, 
  DatabaseStats, 
  DocumentUpload, 
  StreamingChunk,
  HybridSearchRequest,
  SourceDocument,
  HierarchicalResponse 
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  async askQuestion(request: QuestionRequest): Promise<ChatResponse> {
    const response = await api.post('/ask', request);
    return response.data;
  },

  async askQuestionWithHierarchy(request: QuestionRequest): Promise<HierarchicalResponse> {
    const response = await api.post('/ask/hierarchy', request);
    return response.data;
  },

  async askQuestionStreaming(
    request: QuestionRequest, 
    onChunk: (chunk: StreamingChunk) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Failed to get response reader');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        // Keep the last incomplete line in the buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonStr = line.slice(6); // Remove 'data: ' prefix
              if (jsonStr.trim()) {
                const chunkData = JSON.parse(jsonStr) as StreamingChunk;
                onChunk(chunkData);
                
                if (chunkData.is_final) {
                  return;
                }
              }
            } catch (e) {
              console.error('Error parsing chunk:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      throw error;
    }
  },

  async hybridSearch(request: HybridSearchRequest): Promise<{
    documents: SourceDocument[];
    query: string;
    search_type: string;
    city_filter?: string;
    category_filter?: string;
    alpha: number;
  }> {
    const response = await api.post('/search/hybrid', request);
    return response.data;
  },

  async uploadDocuments(documents: DocumentUpload[]): Promise<{ message: string }> {
    const response = await api.post('/documents', documents);
    return response.data;
  },

  async searchDocuments(
    query: string, 
    city?: string, 
    category?: string,
    limit: number = 5,
    searchType: string = 'vector'
  ): Promise<{
    documents: SourceDocument[];
    query: string;
    city_filter?: string;
    category_filter?: string;
    search_type: string;
  }> {
    const params = new URLSearchParams({ 
      query, 
      limit: limit.toString(),
      search_type: searchType
    });
    
    if (city) params.append('city', city);
    if (category) params.append('category', category);
    
    const response = await api.get(`/search?${params}`);
    return response.data;
  },

  async getDatabaseStats(): Promise<DatabaseStats> {
    const response = await api.get('/stats');
    return response.data;
  },

  async healthCheck(): Promise<{ 
    status: string; 
    database_connected: boolean; 
    total_documents: number;
    hybrid_search_enabled?: boolean;
    reranking_enabled?: boolean;
    features?: {
      vector_search: boolean;
      keyword_search: boolean;
      hybrid_search: boolean;
      reranking: boolean;
      streaming: boolean;
      category_filtering: boolean;
    };
  }> {
    const response = await api.get('/health');
    return response.data;
  },
};

export default apiService; 