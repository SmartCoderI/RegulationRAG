import React, { useState, useRef, useEffect } from 'react';
import { Send, Settings, Zap, Search, Filter } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (
    message: string, 
    city?: string, 
    category?: string,
    searchType?: string,
    useReranking?: boolean,
    useStreaming?: boolean,
    useHierarchy?: boolean
  ) => void;
  disabled?: boolean;
  availableCities: string[];
  availableCategories: string[];
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

const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled,
  availableCities,
  availableCategories,
  features
}) => {
  const [message, setMessage] = useState('');
  const [selectedCity, setSelectedCity] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [searchType, setSearchType] = useState<string>('hybrid');
  const [useReranking, setUseReranking] = useState(true);
  const [useStreaming, setUseStreaming] = useState(true);
  const [useHierarchy, setUseHierarchy] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [message]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(
        message.trim(),
        selectedCity || undefined,
        selectedCategory || undefined,
        searchType,
        useReranking,
        useStreaming,
        useHierarchy
      );
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const getSearchTypeColor = (type: string) => {
    switch (type) {
      case 'hybrid': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'vector': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'keyword': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="border-t border-gray-200 bg-white p-4">
      {/* Advanced Options */}
      {showAdvanced && (
        <div className="mb-4 p-4 bg-gray-50 rounded-lg border">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* City Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Filter className="w-4 h-4 inline mr-1" />
                City Filter
              </label>
              <select
                value={selectedCity}
                onChange={(e) => setSelectedCity(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="">All Cities</option>
                {availableCities.map(city => (
                  <option key={city} value={city}>{city}</option>
                ))}
              </select>
            </div>

            {/* Category Filter */}
            {features?.category_filtering && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Category Filter
                </label>
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                >
                  <option value="">All Categories</option>
                  {availableCategories.map(category => (
                    <option key={category} value={category}>{category}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Search Type */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                <Search className="w-4 h-4 inline mr-1" />
                Search Type
              </label>
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value)}
                className={`w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm ${getSearchTypeColor(searchType)}`}
              >
                {features?.hybrid_search && (
                  <option value="hybrid">🔮 Hybrid (Best)</option>
                )}
                {features?.vector_search && (
                  <option value="vector">📊 Vector Semantic</option>
                )}
                {features?.keyword_search && (
                  <option value="keyword">🔍 Keyword</option>
                )}
              </select>
            </div>

            {/* Options */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Options
              </label>
              <div className="space-y-2">
                {features?.reranking && (
                  <label className="flex items-center text-sm">
                    <input
                      type="checkbox"
                      checked={useReranking}
                      onChange={(e) => setUseReranking(e.target.checked)}
                      className="mr-2 rounded"
                    />
                    <Zap className="w-3 h-3 mr-1" />
                    Reranking
                  </label>
                )}
                {features?.streaming && (
                  <label className="flex items-center text-sm">
                    <input
                      type="checkbox"
                      checked={useStreaming}
                      onChange={(e) => setUseStreaming(e.target.checked)}
                      className="mr-2 rounded"
                    />
                    <Send className="w-3 h-3 mr-1" />
                    Streaming
                  </label>
                )}
                {features?.hierarchy && (
                  <label className="flex items-center text-sm">
                    <input
                      type="checkbox"
                      checked={useHierarchy}
                      onChange={(e) => setUseHierarchy(e.target.checked)}
                      className="mr-2 rounded"
                    />
                    <Zap className="w-3 h-3 mr-1" />
                    Hierarchy
                  </label>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Input Area */}
      <form onSubmit={handleSubmit} className="flex items-end space-x-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about city regulations, zoning, parking requirements..."
            disabled={disabled}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none min-h-[48px] max-h-32 disabled:bg-gray-100 disabled:cursor-not-allowed"
            rows={1}
          />
          
          {/* Active Filters Display */}
          {(selectedCity || selectedCategory || searchType !== 'hybrid') && (
            <div className="absolute -top-8 left-0 flex flex-wrap gap-1">
              {selectedCity && (
                <span className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded">
                  📍 {selectedCity}
                </span>
              )}
              {selectedCategory && (
                <span className="inline-flex items-center px-2 py-1 text-xs bg-green-100 text-green-800 rounded">
                  📂 {selectedCategory}
                </span>
              )}
              {searchType !== 'hybrid' && (
                <span className={`inline-flex items-center px-2 py-1 text-xs rounded ${getSearchTypeColor(searchType)}`}>
                  {searchType === 'vector' ? '📊' : searchType === 'keyword' ? '🔍' : '🔮'} {searchType}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={`p-2 rounded-lg transition-colors ${
              showAdvanced 
                ? 'bg-blue-100 text-blue-600 hover:bg-blue-200' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
            title="Advanced Options"
          >
            <Settings className="w-5 h-5" />
          </button>

          <button
            type="submit"
            disabled={!message.trim() || disabled}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center space-x-2"
          >
            <Send className="w-4 h-4" />
            <span>Send</span>
          </button>
        </div>
      </form>

      {/* Quick Actions */}
      {!showAdvanced && (
        <div className="mt-3 flex flex-wrap gap-2">
          {availableCategories.slice(0, 6).map(category => {
            // Create friendly display names and example questions for each category
            const getCategoryInfo = (cat: string) => {
              switch (cat) {
                case 'zoning':
                  return { 
                    label: 'Zoning',
                    question: 'What are the minimum lot size requirements for residential development?' 
                  };
                case 'design_guidelines':
                  return { 
                    label: 'Design Guidelines',
                    question: 'What are the design requirements for Eichler homes?' 
                  };
                case 'street_standards':
                  return { 
                    label: 'Street Standards',
                    question: 'How wide must local streets be?' 
                  };
                case 'utilities':
                  return { 
                    label: 'Utilities',
                    question: 'What are the water pressure requirements for new developments?' 
                  };
                case 'environmental':
                  return { 
                    label: 'Environmental',
                    question: 'What are the stormwater management requirements?' 
                  };
                case 'historic_preservation':
                  return { 
                    label: 'Historic Preservation',
                    question: 'What are the preservation requirements for historic buildings?' 
                  };
                default:
                  return { 
                    label: cat.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                    question: `What are the ${cat.replace(/_/g, ' ')} requirements?` 
                  };
              }
            };

            const { label, question } = getCategoryInfo(category);
            
            return (
              <button
                key={category}
                onClick={() => setMessage(question)}
                className="text-sm px-3 py-1 bg-gray-100 text-gray-600 rounded-full hover:bg-gray-200 transition-colors"
              >
                {label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default ChatInput; 