import React from 'react';
import { Bot, AlertTriangle, CheckCircle, Building, FileText, ExternalLink } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { HierarchicalResponse, SourceDocument, EnhancedSourceDocument } from '../types';

interface HierarchicalChatMessageProps {
  response: HierarchicalResponse;
}

const HierarchicalChatMessage: React.FC<HierarchicalChatMessageProps> = ({ response }) => {
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getJurisdictionIcon = (level?: string) => {
    switch (level) {
      case 'district':
        return '🏛️';
      case 'city':
        return '🏙️';
      case 'county':
        return '🏞️';
      case 'state':
        return '🌟';
      default:
        return '📄';
    }
  };

  const getJurisdictionColor = (level?: string) => {
    switch (level) {
      case 'district':
        return 'bg-purple-50 border-purple-200';
      case 'city':
        return 'bg-blue-50 border-blue-200';
      case 'county':
        return 'bg-green-50 border-green-200';
      case 'state':
        return 'bg-yellow-50 border-yellow-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  const renderSourceSection = (
    title: string, 
    sources: SourceDocument[], 
    isControlling: boolean = false
  ) => {
    if (!sources || sources.length === 0) return null;

    return (
      <div className="mt-4">
        <h4 className={`text-sm font-semibold mb-2 flex items-center ${
          isControlling ? 'text-purple-700' : 'text-gray-700'
        }`}>
          {isControlling && <CheckCircle className="w-4 h-4 mr-1" />}
          {title}
        </h4>
        <div className="space-y-3">
          {sources.map((source, index) => {
            // Cast to enhanced source document to access jurisdiction fields
            const enhancedSource = source as EnhancedSourceDocument;
            
            return (
              <div 
                key={index} 
                className={`border rounded-lg p-3 ${
                  isControlling 
                    ? 'bg-purple-50 border-purple-200' 
                    : getJurisdictionColor(enhancedSource.jurisdiction_level)
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-lg">
                      {getJurisdictionIcon(enhancedSource.jurisdiction_level)}
                    </span>
                    <div>
                      <h5 className="font-medium text-gray-900 text-sm">
                        {source.title || 'Unknown Document'}
                      </h5>
                      <p className="text-xs text-gray-600">
                        {enhancedSource.jurisdiction_name || source.city || 'Unknown Jurisdiction'}
                        {enhancedSource.controlling_authority && (
                          <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800">
                            <CheckCircle className="w-3 h-3 mr-1" />
                            Controls
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                  {source.source_url && (
                    <a
                      href={source.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-600 hover:text-blue-800"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  )}
                </div>
                
                <div className="text-sm text-gray-700 bg-white rounded p-2 border">
                  <ReactMarkdown className="prose prose-sm max-w-none">
                    {source.content.length > 300 
                      ? `${source.content.substring(0, 300)}...` 
                      : source.content}
                  </ReactMarkdown>
                </div>

                {enhancedSource.conflicts_with && enhancedSource.conflicts_with.length > 0 && (
                  <div className="mt-2 flex items-center text-xs text-amber-700 bg-amber-50 rounded px-2 py-1">
                    <AlertTriangle className="w-3 h-3 mr-1" />
                    Superseded by: {enhancedSource.conflicts_with.join(', ')}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="flex justify-start">
      <div className="flex items-start space-x-3 max-w-5xl">
        <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
          <Bot className="w-4 h-4 text-white" />
        </div>
        
        <div className="bg-white border border-gray-200 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
          {/* Conflict Detection Alert */}
          {response.conflicts_detected && (
            <div className="mb-4 flex items-center space-x-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-amber-600" />
              <div>
                <p className="text-sm font-medium text-amber-800">Regulatory Conflicts Detected</p>
                <p className="text-xs text-amber-700">Multiple jurisdictions have different requirements. The most specific regulation controls.</p>
              </div>
            </div>
          )}

          {/* Main Response */}
          <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{response.answer}</ReactMarkdown>
          </div>

          {/* Hierarchy Explanation */}
          {response.hierarchy_explanation && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="text-sm font-semibold text-blue-800 mb-2 flex items-center">
                <Building className="w-4 h-4 mr-1" />
                Regulatory Hierarchy
              </h4>
              <div className="text-sm text-blue-700">
                <ReactMarkdown className="prose prose-sm max-w-none">
                  {response.hierarchy_explanation}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* Controlling Regulations */}
          {renderSourceSection(
            "Controlling Regulations", 
            response.controlling_regulations, 
            true
          )}

          {/* Supporting Regulations */}
          {renderSourceSection(
            "Supporting/Reference Regulations", 
            response.supporting_regulations, 
            false
          )}

          {/* Response Metadata */}
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500 space-y-1">
            <div className="flex items-center space-x-4">
              <span>Search: {response.search_type}</span>
              {response.city_filter && <span>City: {response.city_filter}</span>}
              {response.category_filter && <span>Category: {response.category_filter}</span>}
              {response.reranking_used && (
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                  AI Reranked
                </span>
              )}
            </div>
          </div>

          {/* Timestamp */}
          <div className="mt-3 text-xs text-gray-500">
            {formatTimestamp(response.timestamp)}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HierarchicalChatMessage; 