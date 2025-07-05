import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, ChevronDown, ChevronUp, ExternalLink, MapPin, Tag, Search, Zap } from 'lucide-react';
import { ChatMessage as ChatMessageType } from '../types';

interface ChatMessageProps {
  message: ChatMessageType;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const [showSources, setShowSources] = useState(false);

  const formatTimestamp = (timestamp: Date) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getSearchTypeIcon = (searchType?: string) => {
    switch (searchType) {
      case 'hybrid': return '🔮';
      case 'vector': return '📊';
      case 'keyword': return '🔍';
      default: return '🔍';
    }
  };

  const getSearchTypeColor = (searchType?: string) => {
    switch (searchType) {
      case 'hybrid': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'vector': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'keyword': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatSourceContent = (content: string) => {
    // Truncate long content and show preview
    const maxLength = 200;
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  const extractCityFromMetadata = (metadata: any) => {
    return metadata?.city || metadata?.City || 'Unknown';
  };

  const extractCategoryFromMetadata = (metadata: any) => {
    return metadata?.category || metadata?.Category || metadata?.section || 'General';
  };

  if (message.type === 'user') {
    return (
      <div className="flex justify-end">
        <div className="flex items-start space-x-3 max-w-3xl">
          <div className="bg-blue-600 text-white px-4 py-3 rounded-2xl rounded-br-md">
            <div className="prose prose-sm text-white max-w-none">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
            
            {/* User Query Info */}
            {(message.city_filter || message.category_filter || message.search_type) && (
              <div className="mt-2 pt-2 border-t border-blue-500 flex flex-wrap gap-1">
                {message.city_filter && (
                  <span className="inline-flex items-center px-2 py-1 text-xs bg-blue-500 rounded">
                    <MapPin className="w-3 h-3 mr-1" />
                    {message.city_filter}
                  </span>
                )}
                {message.category_filter && (
                  <span className="inline-flex items-center px-2 py-1 text-xs bg-blue-500 rounded">
                    <Tag className="w-3 h-3 mr-1" />
                    {message.category_filter}
                  </span>
                )}
                {message.search_type && message.search_type !== 'hybrid' && (
                  <span className="inline-flex items-center px-2 py-1 text-xs bg-blue-500 rounded">
                    <Search className="w-3 h-3 mr-1" />
                    {message.search_type}
                  </span>
                )}
                {message.reranking_used && (
                  <span className="inline-flex items-center px-2 py-1 text-xs bg-blue-500 rounded">
                    <Zap className="w-3 h-3 mr-1" />
                    Reranked
                  </span>
                )}
              </div>
            )}
          </div>
          <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
            <User className="w-4 h-4 text-white" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="flex items-start space-x-3 max-w-4xl">
        <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
          <Bot className="w-4 h-4 text-white" />
        </div>
        
        <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
          {/* Response Content */}
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>

          {/* Streaming Indicator */}
          {message.is_streaming && (
            <div className="mt-2 flex items-center text-xs text-gray-500">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse mr-2"></div>
              Generating response...
            </div>
          )}

          {/* Response Metadata */}
          {(message.search_type || message.reranking_used || message.city_filter || message.category_filter) && (
            <div className="mt-3 pt-3 border-t border-gray-200">
              <div className="flex flex-wrap gap-2 text-xs">
                {message.search_type && (
                  <span className={`inline-flex items-center px-2 py-1 rounded border ${getSearchTypeColor(message.search_type)}`}>
                    {getSearchTypeIcon(message.search_type)} {message.search_type} search
                  </span>
                )}
                {message.reranking_used && (
                  <span className="inline-flex items-center px-2 py-1 bg-yellow-100 text-yellow-800 border border-yellow-200 rounded">
                    <Zap className="w-3 h-3 mr-1" />
                    Reranked
                  </span>
                )}
                {message.city_filter && (
                  <span className="inline-flex items-center px-2 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded">
                    <MapPin className="w-3 h-3 mr-1" />
                    {message.city_filter}
                  </span>
                )}
                {message.category_filter && (
                  <span className="inline-flex items-center px-2 py-1 bg-green-50 text-green-700 border border-green-200 rounded">
                    <Tag className="w-3 h-3 mr-1" />
                    {message.category_filter}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Sources Section */}
          {message.sources && message.sources.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <button
                onClick={() => setShowSources(!showSources)}
                className="flex items-center justify-between w-full text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
              >
                <span>Sources ({message.sources.length})</span>
                {showSources ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </button>

              {showSources && (
                <div className="mt-3 space-y-3">
                  {message.sources.map((source, index) => (
                    <div key={index} className="bg-gray-50 border border-gray-200 rounded-lg p-3">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            {source.title && (
                              <h4 className="text-sm font-medium text-gray-900">
                                {source.title}
                              </h4>
                            )}
                            {source.source_url && (
                              <a
                                href={source.source_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center text-blue-600 hover:text-blue-800 transition-colors"
                                title="Open source"
                              >
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            )}
                          </div>
                          
                          <div className="flex flex-wrap gap-1 mb-2">
                            {source.city && (
                              <span className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                                <MapPin className="w-3 h-3 mr-1" />
                                {source.city}
                              </span>
                            )}
                            {source.category && (
                              <span className="inline-flex items-center px-2 py-1 text-xs bg-green-100 text-green-800 rounded">
                                <Tag className="w-3 h-3 mr-1" />
                                {source.category}
                              </span>
                            )}
                            <span className="inline-flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded">
                              Score: {source.score ? (source.score * 100).toFixed(0) : 'N/A'}%
                            </span>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-sm text-gray-600 leading-relaxed">
                        {formatSourceContent(source.content)}
                      </div>
                      
                      {source.metadata && Object.keys(source.metadata).length > 0 && (
                        <details className="mt-2">
                          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                            View metadata
                          </summary>
                          <div className="mt-1 text-xs text-gray-500 bg-gray-100 p-2 rounded">
                            <pre className="whitespace-pre-wrap">
                              {JSON.stringify(source.metadata, null, 2)}
                            </pre>
                          </div>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Timestamp */}
          <div className="mt-3 text-xs text-gray-500">
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage; 