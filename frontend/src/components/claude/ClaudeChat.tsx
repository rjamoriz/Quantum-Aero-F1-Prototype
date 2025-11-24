/**
 * Claude Chat Component
 * Natural language interface for aerodynamic optimization
 */
import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useAnthropic } from '../../hooks/useAnthropic';
import './ClaudeChat.css';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    mesh_id?: string;
    confidence?: number;
    agents_involved?: string[];
  };
}

interface ClaudeChatProps {
  initialContext?: {
    mesh_id?: string;
    parameters?: Record<string, any>;
  };
  onMeshSelect?: (mesh_id: string) => void;
}

export const ClaudeChat: React.FC<ClaudeChatProps> = ({
  initialContext,
  onMeshSelect
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `👋 Hi! I'm Claude, your AI aerodynamics assistant for the Q-Aero F1 optimization system.

I can help you with:
- **Aerodynamic analysis** - Analyze wing designs, flow patterns, and performance
- **Optimization** - Find optimal configurations for specific tracks
- **Trade-off analysis** - Balance downforce, drag, and flutter margins
- **Quick predictions** - Fast ML-based estimates
- **Design recommendations** - Suggest modifications based on CFD/ML results

What would you like to work on today?`,
      timestamp: new Date()
    }
  ]);

  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { streamMessage, isLoading, error } = useAnthropic();

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const trimmedInput = input.trim();

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: trimmedInput,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsStreaming(true);

    try {
      // Create placeholder for assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: '',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Stream response
      const stream = await streamMessage(
        [...messages, userMessage],
        initialContext
      );

      let fullResponse = '';
      for await (const chunk of stream) {
        fullResponse += chunk;

        // Update last message (assistant) with accumulated response
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: fullResponse
          };
          return updated;
        });
      }

    } catch (err) {
      console.error('Chat error:', err);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: '❌ Sorry, I encountered an error. Please try again.',
          timestamp: new Date()
        }
      ]);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClearChat = () => {
    setMessages([{
      role: 'assistant',
      content: 'Chat cleared. How can I help you?',
      timestamp: new Date()
    }]);
  };

  return (
    <div className="claude-chat-container">
      {/* Header */}
      <div className="chat-header">
        <div className="header-left">
          <div className="claude-logo">🤖</div>
          <div className="header-info">
            <h3>Claude AI Assistant</h3>
            <p className="status">
              {isStreaming ? (
                <><span className="status-dot streaming"></span> Thinking...</>
              ) : (
                <><span className="status-dot ready"></span> Ready</>
              )}
            </p>
          </div>
        </div>
        <div className="header-actions">
          <button
            className="btn-secondary"
            onClick={handleClearChat}
            title="Clear chat"
          >
            🗑️ Clear
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              <div className="message-header">
                <span className="message-role">
                  {msg.role === 'user' ? 'You' : 'Claude'}
                </span>
                <span className="message-time">
                  {msg.timestamp.toLocaleTimeString()}
                </span>
              </div>
              <div className="message-body">
                <ReactMarkdown
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '');
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      );
                    }
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              </div>
              {msg.metadata && (
                <div className="message-metadata">
                  {msg.metadata.mesh_id && (
                    <span className="metadata-tag">
                      📐 {msg.metadata.mesh_id}
                    </span>
                  )}
                  {msg.metadata.confidence !== undefined && (
                    <span className="metadata-tag">
                      🎯 {(msg.metadata.confidence * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
        {isStreaming && messages[messages.length - 1]?.content === '' && (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="chat-input-area">
        {error && (
          <div className="error-banner">
            ⚠️ {error}
          </div>
        )}

        {initialContext?.mesh_id && (
          <div className="context-banner">
            📐 Current mesh: <strong>{initialContext.mesh_id}</strong>
          </div>
        )}

        <div className="input-container">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about aerodynamics, optimization, or analysis..."
            disabled={isStreaming}
            rows={3}
          />
          <button
            className="btn-send"
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
          >
            {isStreaming ? '⏳' : '📤'} {isStreaming ? 'Thinking...' : 'Send'}
          </button>
        </div>

        <div className="input-hints">
          <span className="hint">💡 Try: "Optimize this wing for Monza"</span>
          <span className="hint">💡 Try: "Analyze pressure distribution"</span>
          <span className="hint">💡 Try: "Compare with baseline"</span>
        </div>
      </div>
    </div>
  );
};

export default ClaudeChat;
