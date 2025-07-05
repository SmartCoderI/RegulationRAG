#!/bin/bash

echo "📄 Bay Area Regulations - Document Ingestion with Ollama"
echo "========================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    echo "❌ Please run this script from the rag_project directory"
    exit 1
fi

# Check if Ollama is installed (instead of OpenAI API key)
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "Please install Ollama first:"
    echo "  1. Visit: https://ollama.ai/"
    echo "  2. Download and install for your platform"
    echo "  3. Start Ollama: ollama serve"
    echo ""
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running."
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    exit 1
fi

echo "✓ Ollama is running"

echo "🔧 Setting up environment..."
cd backend

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run ./start.sh first."
    exit 1
fi

# Install dependencies
echo "📚 Installing dependencies..."
pip install -q PyPDF2 langchain-ollama ollama

echo "🤖 Checking required Ollama models..."
# Check if required models are available
if ! ollama list | grep -q "llama2"; then
    echo "📥 Downloading llama2 model (this may take a few minutes)..."
    ollama pull llama2
fi

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "📥 Downloading nomic-embed-text model (for embeddings)..."
    ollama pull nomic-embed-text
fi

echo "✓ All required models are available"
echo ""

echo "📄 Processing your documents..."
echo ""

# Run the ingestion script
python scripts/ingest_user_docs.py

echo ""
echo "✅ Document processing complete!"
echo "Your regulations are now searchable in the chat interface."
echo "🎉 All processing was done locally with Ollama - no API costs!" 