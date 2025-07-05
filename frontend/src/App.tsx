import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, Database, AlertCircle, CheckCircle, Zap, Search } from 'lucide-react';
import ChatMessage from './components/ChatMessage';
import HierarchicalChatMessage from './components/HierarchicalChatMessage';
import ChatInput from './components/ChatInput';
import TypingIndicator from './components/TypingIndicator';
import { ChatMessage as ChatMessageType, DatabaseStats, StreamingChunk, HierarchicalResponse } from './types';
import { apiService } from './services/api';
import { v4 as uuidv4 } from 'uuid';

// Enhanced message type that can handle both regular and hierarchical responses
interface EnhancedChatMessage extends ChatMessageType {
  hierarchical_response?: HierarchicalResponse;
  is_hierarchical?: boolean;
}

function App() {
  const [messages, setMessages] = useState<EnhancedChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<DatabaseStats | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Check connection and load stats on mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const healthCheck = await apiService.healthCheck();
        if (healthCheck.status === 'healthy') {
          setConnectionStatus('connected');
          const statsData = await apiService.getDatabaseStats();
          setStats(statsData);
        } else {
          setConnectionStatus('error');
        }
      } catch (error) {
        console.error('Connection check failed:', error);
        setConnectionStatus('error');
      }
    };

    checkConnection();
  }, []);

  const handleSendMessage = async (
    messageText: string, 
    city?: string, 
    category?: string,
    searchType?: string,
    useReranking?: boolean,
    useStreaming?: boolean,
    useHierarchy?: boolean
  ) => {
    setError(null);
    
    // Add user message
    const userMessage: EnhancedChatMessage = {
      id: uuidv4(),
      type: 'user',
      content: messageText,
      timestamp: new Date(),
      city_filter: city,
      category_filter: category,
      search_type: searchType,
      reranking_used: useReranking,
    };
    
    setMessages(prev => [...prev, userMessage]);

    // Prepare request
    const request = {
      question: messageText,
      city,
      category,
      context_limit: 5,
      search_type: (searchType as "hybrid" | "vector" | "keyword") || 'hybrid',
      use_reranking: useReranking ?? true,
      stream: useStreaming ?? false,
    };

    try {
      // Check if hierarchical query is requested
      if (useHierarchy) {
        // Handle hierarchical response (non-streaming for now)
        setIsLoading(true);
        const hierarchicalResponse = await apiService.askQuestionWithHierarchy(request);
        
        // Add hierarchical assistant response
        const assistantMessage: EnhancedChatMessage = {
          id: uuidv4(),
          type: 'assistant',
          content: hierarchicalResponse.answer,
          timestamp: new Date(hierarchicalResponse.timestamp),
          sources: hierarchicalResponse.sources,
          city_filter: hierarchicalResponse.city_filter,
          category_filter: hierarchicalResponse.category_filter,
          search_type: hierarchicalResponse.search_type,
          reranking_used: hierarchicalResponse.reranking_used,
          hierarchical_response: hierarchicalResponse,
          is_hierarchical: true,
        };

        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
        
      } else if (useStreaming) {
        // Handle streaming response
        setIsLoading(true);
        
        // Create initial assistant message for streaming
        const assistantMessageId = uuidv4();
        const initialAssistantMessage: EnhancedChatMessage = {
          id: assistantMessageId,
          type: 'assistant',
          content: '',
          timestamp: new Date(),
          is_streaming: true,
          search_type: searchType,
          city_filter: city,
          category_filter: category,
          reranking_used: useReranking,
        };
        
        setMessages(prev => [...prev, initialAssistantMessage]);
        setStreamingMessageId(assistantMessageId);

        let accumulatedContent = '';
        
        await apiService.askQuestionStreaming(request, (chunk: StreamingChunk) => {
          if (chunk.is_final) {
            // Final chunk with sources - use accumulated content + final chunk
            const finalContent = accumulatedContent + chunk.chunk;
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? {
                    ...msg,
                    content: finalContent,
                    sources: chunk.sources,
                    is_streaming: false,
                    timestamp: new Date(),
                  }
                : msg
            ));
            setStreamingMessageId(null);
            setIsLoading(false);
          } else {
            // Intermediate chunk
            accumulatedContent += chunk.chunk;
            setMessages(prev => prev.map(msg => 
              msg.id === assistantMessageId 
                ? { ...msg, content: accumulatedContent }
                : msg
            ));
          }
        });

      } else {
        // Handle regular response
        setIsLoading(true);
        const response = await apiService.askQuestion(request);

        // Add assistant response
        const assistantMessage: EnhancedChatMessage = {
          id: uuidv4(),
          type: 'assistant',
          content: response.answer,
          timestamp: new Date(response.timestamp),
          sources: response.sources,
          city_filter: response.city_filter,
          category_filter: response.category_filter,
          search_type: response.search_type,
          reranking_used: response.reranking_used,
        };

        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
      }

    } catch (error: any) {
      console.error('Error sending message:', error);
      setError(error.response?.data?.detail || 'Failed to get response. Please try again.');
      
      // Add error message
      const errorMessage: EnhancedChatMessage = {
        id: uuidv4(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while processing your question. Please try again or rephrase your question.',
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
      setStreamingMessageId(null);
    }
  };

  const getWelcomeMessage = (): EnhancedChatMessage => {
    const featuresText = stats?.features ? `

## 🚀 Enhanced Features Available:
${stats.features.hybrid_search ? '- **Hybrid Search**: Combines semantic and keyword search for best results' : ''}
${stats.features.reranking ? '- **AI Reranking**: Cohere-powered relevance optimization' : ''}
${stats.features.streaming ? '- **Streaming Responses**: Real-time answer generation' : ''}
${stats.features.category_filtering ? '- **Category Filtering**: Filter by regulation type (zoning, parking, etc.)' : ''}
${stats.features.vector_search ? '- **Vector Search**: Semantic similarity search' : ''}
${stats.features.keyword_search ? '- **Keyword Search**: Traditional text matching' : ''}
` : '';

    // Reorder cities to match preferred order: San Francisco, Sunnyvale, California, General
    const getOrderedCities = () => {
      return 'San Francisco, Sunnyvale, Santa Clara County, California, and General';
    };

    return {
      id: 'welcome',
      type: 'assistant',
      content: `# Welcome to Bay Area Land Development Regulations Assistant! 😊

I'm here to help you navigate land development regulations across Bay Area cities. I currently have information about **${getOrderedCities()}** with **${stats?.total_documents || 0}** documents covering **${stats?.categories.join(', ') || 'various categories'}**.

## 💡 What I can help with:
- **Zoning requirements** (lot sizes, setbacks, height limits)
- **Parking standards** (ratios, requirements)
- **Street design** (widths, sidewalks, landscaping)
- **Development guidelines** (open space, unit mix)
- **Infrastructure requirements** (utilities, stormwater)
${featuresText}

## 🎯 Try asking:
- "What's the minimum lot frontage for a duplex in San Francisco?"
- "How wide must local streets be in Sunnyvale?"
- "What are the parking requirements for multi-family housing?"

Use the **⚙️ Advanced Options** below to filter by city/category and customize search settings!`,
      timestamp: new Date(),
    };
  };

  // Show welcome message if no messages yet
  const displayMessages = messages.length === 0 ? [getWelcomeMessage()] : messages;

  const getFeaturesBadge = () => {
    if (!stats?.features) return null;
    
    const activeFeatures = Object.entries(stats.features).filter(([_, enabled]) => enabled).length;
    const totalFeatures = Object.keys(stats.features).length;
    
    return (
      <div className="flex items-center space-x-2 text-sm text-gray-600">
        <Zap className="w-4 h-4 text-yellow-500" />
        <span>{activeFeatures}/{totalFeatures} features enabled</span>
      </div>
    );
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary-500 rounded-lg flex items-center justify-center">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">
                Bay Area Land Development Regulations Assistant
              </h1>
              <p className="text-sm text-gray-600">
                {stats?.features?.hybrid_search ? 'Enhanced AI-powered' : 'AI-powered'} city regulations assistant
              </p>
            </div>
          </div>
          
          {/* Status and Stats */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' : 
                connectionStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500'
              }`} />
              <span className="text-sm text-gray-600">
                {connectionStatus === 'connected' ? 'Connected' : 
                 connectionStatus === 'error' ? 'Connection Error' : 'Connecting...'}
              </span>
            </div>
            
            {stats && (
              <>
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Database className="w-4 h-4" />
                  <span>{stats.total_documents} documents</span>
                  {stats.total_chunks && (
                    <span className="text-gray-400">({stats.total_chunks} chunks)</span>
                  )}
                  <span>•</span>
                  <span>{stats.cities.length} cities</span>
                  <span>•</span>
                  <span>{stats.categories?.length || 0} categories</span>
                </div>
                {getFeaturesBadge()}
              </>
            )}
          </div>
        </div>
      </header>

      {/* Bay Area Banner */}
      <div className="relative h-32 bg-gradient-to-r from-blue-500 to-orange-400 overflow-hidden">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-90"
          style={{
            backgroundImage: `url('https://images.unsplash.com/photo-1449824913935-59a10b8d2000?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80')`
          }}
        />
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/30 to-orange-900/30" />
        <div className="relative h-full flex items-center justify-center">
          <div className="text-center text-white">
            <h2 className="text-2xl font-bold drop-shadow-lg">
              Bay Area Land Development Regulations
            </h2>
            <p className="text-sm opacity-90 drop-shadow">
              Navigate city planning requirements across San Francisco, Sunnyvale & Beyond
            </p>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 border-b border-red-200 px-6 py-3">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-red-600" />
            <span className="text-sm text-red-800">{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-600 hover:text-red-700"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto chat-container px-6 py-4 space-y-4">
        {displayMessages.map((message) => (
          message.is_hierarchical && message.hierarchical_response ? (
            <HierarchicalChatMessage key={message.id} response={message.hierarchical_response} />
          ) : (
            <ChatMessage key={message.id} message={message} />
          )
        ))}
        
        {isLoading && !streamingMessageId && <TypingIndicator />}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <ChatInput
        onSendMessage={handleSendMessage}
        disabled={isLoading || connectionStatus === 'error'}
        availableCities={stats?.cities || ['San Francisco', 'Sunnyvale']}
        availableCategories={stats?.categories || ['zoning', 'parking', 'street_standards']}
        features={stats?.features}
      />
    </div>
  );
}

export default App; 