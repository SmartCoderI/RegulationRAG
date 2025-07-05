#!/usr/bin/env python3
"""
Enhanced Resumable Document Ingestion Script

This script provides resumable document ingestion with:
- Progress tracking and checkpointing
- Resume from interruptions
- Detailed progress reporting
- Error recovery and retry logic
- Support for both sample and user documents
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import signal
import time

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.resumable_ingestion import ResumableIngestionService, DocumentCheckpoint
from app.services.rag_service import EnhancedRAGService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingestion.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variable for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown"""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Shutdown requested. Finishing current document...")

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def load_sample_documents() -> List[Dict]:
    """Load sample regulation documents"""
    from ingest_data import load_sample_documents as load_samples
    return load_samples()

def load_user_documents(source_dir: str) -> List[Dict]:
    """Load user documents from directory"""
    from ingest_user_docs import load_user_documents as load_user_docs
    return load_user_docs(source_dir)

def load_pdf_documents(pdf_dir: str) -> List[Dict]:
    """Load PDF documents from a directory"""
    documents = []
    pdf_path = Path(pdf_dir)
    
    if not pdf_path.exists():
        logger.error(f"PDF directory {pdf_dir} does not exist")
        return documents
    
    logger.info(f"Loading PDF documents from {pdf_dir}")
    
    try:
        import PyPDF2
        PDF_AVAILABLE = True
    except ImportError:
        try:
            from pypdf import PdfReader
            PDF_AVAILABLE = True
        except ImportError:
            PDF_AVAILABLE = False
            logger.error("PDF parsing libraries not available. Please install PyPDF2 or pypdf.")
            return documents
    
    def extract_text_from_pdf(pdf_file: Path) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(pdf_file, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                except:
                    from pypdf import PdfReader
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_file}: {str(e)}")
            return ""
    
    # Process each PDF file
    for pdf_file in pdf_path.glob("*.pdf"):
        try:
            logger.info(f"Processing: {pdf_file.name}")
            
            content = extract_text_from_pdf(pdf_file)
            if not content:
                logger.warning(f"No text extracted from {pdf_file.name}")
                continue
            
            # Simple categorization
            filename_lower = pdf_file.name.lower()
            if "sf" in filename_lower or "san francisco" in filename_lower:
                city = "San Francisco"
            elif "sunnyvale" in filename_lower:
                city = "Sunnyvale"
            else:
                city = "General"
            
            if "zoning" in filename_lower:
                category = "zoning"
            elif "street" in filename_lower or "construction" in filename_lower:
                category = "street_standards"
            elif "design" in filename_lower:
                category = "design_guidelines"
            else:
                category = "general"
            
            doc = {
                'title': f"{city} - {pdf_file.stem.replace('_', ' ').replace('-', ' ')}",
                'content': content,
                'city': city,
                'category': category,
                'source_url': f"file://{pdf_file.absolute()}",
                'metadata': {
                    'filename': pdf_file.name,
                    'file_path': str(pdf_file),
                    'file_size': pdf_file.stat().st_size,
                    'pages': content.count('\f') + 1  # Form feed character indicates page breaks
                }
            }
            
            documents.append(doc)
            logger.info(f"✓ Loaded: {doc['title']} ({len(content)} chars)")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {str(e)}")
            continue
    
    return documents

class ProgressReporter:
    """Real-time progress reporting"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_update = time.time()
    
    def on_document_start(self, checkpoint: DocumentCheckpoint):
        """Called when document processing starts"""
        logger.info(f"📄 Starting: {checkpoint.title}")
    
    def on_document_complete(self, checkpoint: DocumentCheckpoint):
        """Called when document processing completes"""
        elapsed = time.time() - self.start_time
        logger.info(f"✅ Completed: {checkpoint.title} (chunks: {checkpoint.chunk_count}, time: {elapsed:.1f}s)")
    
    def on_document_failed(self, checkpoint: DocumentCheckpoint):
        """Called when document processing fails"""
        logger.error(f"❌ Failed: {checkpoint.title} - {checkpoint.error_message}")
    
    def on_progress_update(self, progress: Dict):
        """Called on progress updates"""
        current_time = time.time()
        if current_time - self.last_update > 5:  # Update every 5 seconds
            self.last_update = current_time
            elapsed = current_time - self.start_time
            
            completed = progress['completed_documents']
            total = progress['total_documents']
            percentage = progress['completion_percentage']
            
            if completed > 0:
                avg_time = elapsed / completed
                remaining = total - completed
                eta = avg_time * remaining
                logger.info(f"📊 Progress: {completed}/{total} ({percentage:.1f}%) - ETA: {eta:.0f}s")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced Resumable Document Ingestion')
    parser.add_argument('--source-type', choices=['sample', 'user', 'pdf'], required=True,
                       help='Type of documents to ingest')
    parser.add_argument('--source-dir', help='Source directory for user/pdf documents')
    parser.add_argument('--session-id', help='Session ID to resume (optional)')
    parser.add_argument('--list-sessions', action='store_true', help='List available sessions')
    parser.add_argument('--cleanup-session', help='Clean up specific session')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed documents in current session')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retries per document')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Directory for checkpoint files')
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    # Initialize resumable ingestion service
    ingestion_service = ResumableIngestionService(
        checkpoint_dir=args.checkpoint_dir,
        max_retries=args.max_retries
    )
    
    # Handle list sessions
    if args.list_sessions:
        sessions = ingestion_service.list_sessions()
        if sessions:
            print("\n📋 Available Sessions:")
            for session in sessions:
                print(f"  • {session['session_id']}: {session['completed_documents']}/{session['total_documents']} docs")
        else:
            print("No sessions found.")
        return
    
    # Handle cleanup
    if args.cleanup_session:
        ingestion_service.progress_file = Path(args.checkpoint_dir) / f"{args.cleanup_session}_progress.json"
        ingestion_service.cleanup_session(keep_logs=False)
        print(f"Cleaned up session: {args.cleanup_session}")
        return
    
    # Load documents based on source type
    documents = []
    
    if args.source_type == 'sample':
        logger.info("Loading sample documents...")
        documents = load_sample_documents()
        
    elif args.source_type == 'user':
        if not args.source_dir:
            logger.error("--source-dir required for user documents")
            sys.exit(1)
        logger.info(f"Loading user documents from {args.source_dir}...")
        documents = load_user_documents(args.source_dir)
        
    elif args.source_type == 'pdf':
        if not args.source_dir:
            logger.error("--source-dir required for PDF documents")
            sys.exit(1)
        logger.info(f"Loading PDF documents from {args.source_dir}...")
        documents = load_pdf_documents(args.source_dir)
    
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
    
    logger.info(f"📊 Document Summary:")
    logger.info(f"   Cities: {dict(cities)}")
    logger.info(f"   Categories: {dict(categories)}")
    
    # Setup progress reporting
    reporter = ProgressReporter()
    ingestion_service.on_document_start = reporter.on_document_start
    ingestion_service.on_document_complete = reporter.on_document_complete
    ingestion_service.on_document_failed = reporter.on_document_failed
    ingestion_service.on_progress_update = reporter.on_progress_update
    
    try:
        # Start or resume ingestion
        session_id = ingestion_service.start_ingestion(documents, args.session_id)
        logger.info(f"🚀 Starting ingestion session: {session_id}")
        
        # Handle retry failed documents
        if args.retry_failed:
            retry_count = ingestion_service.retry_failed_documents()
            logger.info(f"🔄 Reset {retry_count} failed documents for retry")
        
        # Check for Ollama
        try:
            service = EnhancedRAGService()
            logger.info("✅ Enhanced RAG service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Could not initialize RAG service: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
            sys.exit(1)
        
        # Ingest documents
        logger.info("🔄 Starting document ingestion...")
        start_time = time.time()
        
        result = ingestion_service.ingest_documents(documents)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Print final results
        logger.info("🎉 Ingestion completed!")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Session ID: {result['session_id']}")
        logger.info(f"   Total Documents: {result['total_documents']}")
        logger.info(f"   Completed: {result['completed_documents']}")
        logger.info(f"   Failed: {result['failed_documents']}")
        logger.info(f"   Skipped: {result['skipped_documents']}")
        logger.info(f"   Success Rate: {result['completion_percentage']:.1f}%")
        logger.info(f"   Total Time: {elapsed:.1f}s")
        
        if result['failed_documents'] > 0:
            failed_docs = ingestion_service.get_failed_documents()
            logger.warning("❌ Failed Documents:")
            for doc in failed_docs:
                logger.warning(f"   • {doc.title}: {doc.error_message}")
            logger.info(f"💡 To retry failed documents: --retry-failed --session-id {session_id}")
        
        # Cleanup if fully successful
        if result['is_complete'] and result['failed_documents'] == 0:
            logger.info("🧹 Cleaning up successful session...")
            ingestion_service.cleanup_session(keep_logs=True)
        
    except KeyboardInterrupt:
        logger.info("⏸️  Ingestion interrupted by user")
        logger.info(f"💾 Progress saved. Resume with: --session-id {session_id}")
        
    except Exception as e:
        logger.error(f"💥 Error during ingestion: {str(e)}")
        if 'session_id' in locals():
            logger.info(f"💾 Progress saved. Resume with: --session-id {session_id}")
        sys.exit(1)

if __name__ == "__main__":
    main() 