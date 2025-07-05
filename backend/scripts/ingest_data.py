#!/usr/bin/env python3
"""
Data ingestion script for Bay Area city regulations.
This script loads regulation documents and adds them to the vector database.
Now uses Ollama for free local processing instead of OpenAI.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import RAGService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents_from_directory(data_dir: str) -> List[Dict]:
    """Load all regulation documents from the data directory"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return documents
    
    # Process each city directory
    for city_dir in data_path.iterdir():
        if not city_dir.is_dir():
            continue
            
        city_name = city_dir.name.replace('_', ' ').title()
        logger.info(f"Processing documents for {city_name}")
        
        # Process each document file in the city directory
        for doc_file in city_dir.iterdir():
            if doc_file.suffix not in ['.txt', '.md', '.pdf']:
                continue
                
            try:
                # Read document content
                if doc_file.suffix == '.pdf':
                    # For PDF files, you'd need to implement PDF parsing
                    logger.warning(f"PDF parsing not implemented for {doc_file}")
                    continue
                else:
                    with open(doc_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                # Determine document category from filename
                filename = doc_file.stem.lower()
                if 'zoning' in filename:
                    category = 'zoning'
                elif 'street' in filename or 'public_works' in filename:
                    category = 'street_standards'
                elif 'parking' in filename:
                    category = 'parking'
                elif 'tree' in filename or 'landscape' in filename:
                    category = 'landscaping'
                else:
                    category = 'general'
                
                # Create document object
                doc = {
                    'title': f"{city_name} - {doc_file.stem.replace('_', ' ').title()}",
                    'content': content,
                    'city': city_name,
                    'category': category,
                    'metadata': {
                        'filename': doc_file.name,
                        'file_path': str(doc_file),
                        'file_size': doc_file.stat().st_size
                    }
                }
                
                documents.append(doc)
                logger.info(f"Loaded document: {doc['title']}")
                
            except Exception as e:
                logger.error(f"Error loading document {doc_file}: {str(e)}")
                continue
    
    return documents

def load_sample_documents() -> List[Dict]:
    """Load the sample regulation documents we created"""
    base_dir = Path(__file__).parent.parent / "data" / "regulations"
    return load_documents_from_directory(str(base_dir))

async def ingest_documents(documents: List[Dict]):
    """Ingest documents into the RAG system"""
    try:
        logger.info(f"Initializing RAG service with Ollama...")
        rag_service = RAGService()
        
        logger.info(f"Adding {len(documents)} documents to vector database...")
        rag_service.add_documents(documents)
        
        # Get stats after ingestion
        stats = rag_service.get_database_stats()
        logger.info(f"Ingestion complete. Database stats: {stats}")
        
    except Exception as e:
        logger.error(f"Error during document ingestion: {str(e)}")
        raise

def main():
    """Main function"""
    logger.info("Starting sample document ingestion process with Ollama...")
    
    # Check for Ollama instead of OpenAI API key
    try:
        import ollama
        models = ollama.list()
        logger.info("✓ Ollama is available")
    except Exception as e:
        logger.warning(f"⚠️  Ollama not available: {e}")
        logger.info("This is fine for testing, but you'll need Ollama for AI features")
        logger.info("Install Ollama: https://ollama.ai/")
        # Don't exit - allow the system to start without AI features
    
    # OpenAI check (commented out - now using Ollama)
    # if not os.getenv("OPENAI_API_KEY"):
    #     logger.error("OPENAI_API_KEY environment variable is required")
    #     sys.exit(1)
    
    try:
        # Load documents
        documents = load_sample_documents()
        
        if not documents:
            logger.warning("No sample documents found to ingest")
            return
        
        logger.info(f"Found {len(documents)} sample documents to ingest")
        
        # Print document summary
        cities = set(doc['city'] for doc in documents)
        categories = set(doc['category'] for doc in documents)
        
        logger.info(f"Cities: {', '.join(sorted(cities))}")
        logger.info(f"Categories: {', '.join(sorted(categories))}")
        
        # Ingest documents (only if Ollama is available)
        try:
            asyncio.run(ingest_documents(documents))
            logger.info("✅ Sample document ingestion completed successfully!")
        except Exception as e:
            logger.warning(f"⚠️  Could not ingest documents (Ollama not ready): {e}")
            logger.info("The system will still start, but you'll need to set up Ollama for AI features")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        # Don't exit with error - allow the system to start
        logger.info("Continuing startup without sample data...")

if __name__ == "__main__":
    main() 