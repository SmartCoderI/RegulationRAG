#!/usr/bin/env python3
"""
User document ingestion script for Bay Area city regulations.
This script loads PDF documents from the user's source directory.
Now uses Ollama for free local processing instead of OpenAI.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import re

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service import RAGService
from dotenv import load_dotenv

# PDF parsing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        from pypdf import PdfReader
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            try:
                # Try PyPDF2 first
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            except:
                # Fallback to pypdf
                from pypdf import PdfReader
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def categorize_document(filename: str, content: str) -> tuple[str, str]:
    """
    Categorize document by city and type based on filename and content
    Returns (city, category)
    """
    filename_lower = filename.lower()
    content_lower = content.lower()
    
    # Determine city
    city = "General"  # default
    if "sf" in filename_lower or "san francisco" in filename_lower:
        city = "San Francisco"
    elif "sunnyvale" in filename_lower:
        city = "Sunnyvale"
    elif "calgreen" in filename_lower:
        city = "California"  # State-level regulations
    
    # Determine category
    category = "general"  # default
    
    # Check filename patterns
    if any(word in filename_lower for word in ["zoning", "zone"]):
        category = "zoning"
    elif any(word in filename_lower for word in ["street", "public_works", "roadway", "construction"]):
        category = "street_standards"
    elif any(word in filename_lower for word in ["design", "guidelines", "residential"]):
        category = "design_guidelines"
    elif any(word in filename_lower for word in ["water", "sewer", "storm", "drain"]):
        category = "utilities"
    elif any(word in filename_lower for word in ["heritage", "historic"]):
        category = "historic_preservation"
    elif any(word in filename_lower for word in ["lighting"]):
        category = "infrastructure"
    elif any(word in filename_lower for word in ["calgreen", "green"]):
        category = "environmental"
    
    return city, category

def load_user_documents(source_dir: str) -> List[Dict]:
    """Load all regulation documents from the user's source directory"""
    documents = []
    source_path = Path(source_dir)
    
    if not source_path.exists():
        logger.error(f"Source directory {source_dir} does not exist")
        return documents
    
    logger.info(f"Processing documents from {source_dir}")
    
    # Process each document file
    for doc_file in source_path.iterdir():
        if doc_file.is_file() and doc_file.suffix.lower() == '.pdf':
            try:
                logger.info(f"Processing: {doc_file.name}")
                
                # Extract text from PDF
                if not PDF_AVAILABLE:
                    logger.error("PDF parsing libraries not available. Please install PyPDF2.")
                    continue
                
                content = extract_text_from_pdf(doc_file)
                
                if not content:
                    logger.warning(f"No text extracted from {doc_file.name}")
                    continue
                
                # Categorize document
                city, category = categorize_document(doc_file.name, content)
                
                # Create clean title
                title = doc_file.stem.replace('_', ' ').replace('-', ' ')
                title = re.sub(r'\s+', ' ', title).strip()
                
                # Create document object
                doc = {
                    'title': f"{city} - {title}",
                    'content': content,
                    'city': city,
                    'category': category,
                    'metadata': {
                        'filename': doc_file.name,
                        'file_path': str(doc_file),
                        'file_size': doc_file.stat().st_size,
                        'pages': len(content.split('\n\n'))  # rough page count
                    }
                }
                
                documents.append(doc)
                logger.info(f"✓ Loaded: {doc['title']} ({city}, {category})")
                
            except Exception as e:
                logger.error(f"Error processing document {doc_file}: {str(e)}")
                continue
    
    return documents

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

def check_ollama_available():
    """Check if Ollama is running and has required models"""
    # Simplified check - just verify Ollama is responding
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama is running and accessible")
            return True
        else:
            logger.error("❌ Ollama is not responding properly")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot connect to Ollama: {e}")
        return False

def main():
    """Main function"""
    logger.info("Starting user document ingestion process with Ollama...")
    
    # Check if Ollama is available (instead of OpenAI API key)
    if not check_ollama_available():
        logger.error("Ollama is not properly set up. Please install and configure Ollama first.")
        sys.exit(1)
    
    # OpenAI check (commented out)
    # if not os.getenv("OPENAI_API_KEY"):
    #     logger.error("OPENAI_API_KEY environment variable is required")
    #     sys.exit(1)
    
    # User's source directory
    source_dir = "/Users/yuqingwu/Documents/CS/RAG Land Development/source documents"
    
    try:
        # Load documents
        documents = load_user_documents(source_dir)
        
        if not documents:
            logger.warning("No documents found to ingest")
            return
        
        logger.info(f"Found {len(documents)} documents to ingest")
        
        # Print document summary
        cities = {}
        categories = {}
        
        for doc in documents:
            city = doc['city']
            category = doc['category']
            
            cities[city] = cities.get(city, 0) + 1
            categories[category] = categories.get(category, 0) + 1
        
        logger.info("Document Summary:")
        logger.info(f"Cities: {dict(cities)}")
        logger.info(f"Categories: {dict(categories)}")
        
        # Show document list
        logger.info("\nDocument List:")
        for i, doc in enumerate(documents, 1):
            logger.info(f"{i:2d}. {doc['title']}")
        
        # Confirm before proceeding
        print(f"\nReady to ingest {len(documents)} documents into the RAG system.")
        print("This will:")
        print("- Create embeddings for all documents using Ollama (FREE!)")
        print("- Store them in the ChromaDB vector database")
        print("- Make them searchable via the chat interface")
        print("- Use local AI models (no API costs)")
        
        response = input("\nProceed with ingestion? (y/n): ")
        if response.lower() != 'y':
            print("Ingestion cancelled.")
            return
        
        # Ingest documents
        asyncio.run(ingest_documents(documents))
        
        logger.info("✅ Document ingestion completed successfully!")
        logger.info("You can now ask questions about these regulations in the chat interface.")
        logger.info("All processing is done locally with Ollama - no API costs!")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 