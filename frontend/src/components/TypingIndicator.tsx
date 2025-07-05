import React from 'react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="chat-message assistant">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-500 text-white flex items-center justify-center text-sm font-medium">
          AI
        </div>
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-sm font-medium text-gray-900">
              Bay Area Regulations Assistant
            </span>
            <span className="text-xs text-gray-500">typing...</span>
          </div>
          <div className="typing-indicator">
            <span>Thinking</span>
            <div className="typing-dots">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator; 