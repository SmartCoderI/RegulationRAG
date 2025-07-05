"""
Resumable Document Ingestion Service

This service provides checkpointing and resumable document ingestion:
- Tracks progress after each document
- Can resume from interruptions
- Handles partial failures gracefully
- Provides detailed progress reporting
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .rag_service import EnhancedRAGService

logger = logging.getLogger(__name__)

class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class DocumentCheckpoint:
    """Checkpoint information for a single document"""
    doc_id: str
    title: str
    file_path: str
    file_hash: str
    status: DocumentStatus
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    chunk_count: Optional[int] = None
    retry_count: int = 0

@dataclass
class IngestionProgress:
    """Overall ingestion progress tracking"""
    session_id: str
    started_at: datetime
    last_updated: datetime
    total_documents: int
    completed_documents: int
    failed_documents: int
    skipped_documents: int
    current_document: Optional[str] = None
    documents: Optional[Dict[str, DocumentCheckpoint]] = None

    def __post_init__(self):
        if self.documents is None:
            self.documents = {}

class ResumableIngestionService:
    """Service for resumable document ingestion with checkpointing"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints", max_retries: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_retries = max_retries
        self.rag_service = None
        self.progress_file = self.checkpoint_dir / "ingestion_progress.json"
        self.progress: Optional[IngestionProgress] = None
        
        # Callbacks for progress reporting
        self.on_document_start: Optional[Callable] = None
        self.on_document_complete: Optional[Callable] = None
        self.on_document_failed: Optional[Callable] = None
        self.on_progress_update: Optional[Callable] = None

    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for change detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return str(Path(file_path).stat().st_mtime)  # Use modification time as fallback

    def _generate_doc_id(self, doc: Dict) -> str:
        """Generate unique document ID"""
        # Use file path + title for unique ID
        content = f"{doc.get('metadata', {}).get('file_path', '')}{doc['title']}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _save_progress(self):
        """Save current progress to checkpoint file"""
        if not self.progress:
            return
            
        try:
            # Convert datetime objects to strings for JSON serialization
            progress_dict = asdict(self.progress)
            progress_dict['started_at'] = self.progress.started_at.isoformat()
            progress_dict['last_updated'] = self.progress.last_updated.isoformat()
            
            # Convert document checkpoints
            for doc_id, checkpoint in progress_dict['documents'].items():
                checkpoint['status'] = checkpoint['status'].value if hasattr(checkpoint['status'], 'value') else checkpoint['status']
                if checkpoint['processed_at']:
                    checkpoint['processed_at'] = checkpoint['processed_at'].isoformat() if hasattr(checkpoint['processed_at'], 'isoformat') else checkpoint['processed_at']
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    def _load_progress(self) -> Optional[IngestionProgress]:
        """Load progress from checkpoint file"""
        try:
            if not self.progress_file.exists():
                return None
                
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            
            # Convert string dates back to datetime objects
            data['started_at'] = datetime.fromisoformat(data['started_at'])
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
            
            # Convert document checkpoints
            documents = {}
            for doc_id, checkpoint_data in data.get('documents', {}).items():
                checkpoint_data['status'] = DocumentStatus(checkpoint_data['status'])
                if checkpoint_data.get('processed_at'):
                    checkpoint_data['processed_at'] = datetime.fromisoformat(checkpoint_data['processed_at'])
                else:
                    checkpoint_data['processed_at'] = None
                documents[doc_id] = DocumentCheckpoint(**checkpoint_data)
            
            data['documents'] = documents
            return IngestionProgress(**data)
            
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return None

    def start_ingestion(self, documents: List[Dict], session_id: Optional[str] = None) -> str:
        """
        Start or resume document ingestion
        
        Args:
            documents: List of document dictionaries to ingest
            session_id: Optional session ID to resume (if None, creates new session)
            
        Returns:
            Session ID for this ingestion
        """
        # Initialize RAG service
        if not self.rag_service:
            logger.info("Initializing Enhanced RAG service...")
            self.rag_service = EnhancedRAGService()

        # Load or create progress
        if session_id:
            self.progress = self._load_progress()
            if not self.progress or self.progress.session_id != session_id:
                logger.warning(f"Could not resume session {session_id}, starting new session")
                self.progress = None

        if not self.progress:
            # Create new ingestion session
            session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            self.progress = IngestionProgress(
                session_id=session_id,
                started_at=datetime.now(),
                last_updated=datetime.now(),
                total_documents=len(documents),
                completed_documents=0,
                failed_documents=0,
                skipped_documents=0,
                documents={}
            )
            
            # Initialize document checkpoints
            for doc in documents:
                doc_id = self._generate_doc_id(doc)
                file_path = doc.get('metadata', {}).get('file_path', '')
                file_hash = self._get_file_hash(file_path) if file_path else ''
                
                if self.progress.documents is None:
                    self.progress.documents = {}
                    
                self.progress.documents[doc_id] = DocumentCheckpoint(
                    doc_id=doc_id,
                    title=doc['title'],
                    file_path=file_path,
                    file_hash=file_hash,
                    status=DocumentStatus.PENDING
                )
            
            logger.info(f"Started new ingestion session: {session_id}")
        else:
            logger.info(f"Resuming ingestion session: {self.progress.session_id}")
            
            # Check for file changes in existing documents
            self._check_file_changes(documents)
            
            # Add any new documents
            self._add_new_documents(documents)

        self._save_progress()
        return self.progress.session_id

    def _check_file_changes(self, documents: List[Dict]):
        """Check if any files have changed since last run"""
        if not self.progress or not self.progress.documents:
            return
            
        for doc in documents:
            doc_id = self._generate_doc_id(doc)
            if doc_id in self.progress.documents:
                checkpoint = self.progress.documents[doc_id]
                file_path = doc.get('metadata', {}).get('file_path', '')
                
                if file_path:
                    current_hash = self._get_file_hash(file_path)
                    if current_hash != checkpoint.file_hash:
                        logger.info(f"File changed, re-processing: {checkpoint.title}")
                        checkpoint.file_hash = current_hash
                        checkpoint.status = DocumentStatus.PENDING
                        checkpoint.retry_count = 0
                        checkpoint.error_message = None

    def _add_new_documents(self, documents: List[Dict]):
        """Add any new documents not in the current session"""
        if not self.progress or not self.progress.documents:
            return
            
        existing_docs = set(self.progress.documents.keys())
        new_count = 0
        
        for doc in documents:
            doc_id = self._generate_doc_id(doc)
            if doc_id not in existing_docs:
                file_path = doc.get('metadata', {}).get('file_path', '')
                file_hash = self._get_file_hash(file_path) if file_path else ''
                
                self.progress.documents[doc_id] = DocumentCheckpoint(
                    doc_id=doc_id,
                    title=doc['title'],
                    file_path=file_path,
                    file_hash=file_hash,
                    status=DocumentStatus.PENDING
                )
                new_count += 1
        
        if new_count > 0:
            self.progress.total_documents += new_count
            logger.info(f"Added {new_count} new documents to session")

    def ingest_documents(self, documents: List[Dict]) -> Dict:
        """
        Ingest documents with resumable checkpointing
        
        Args:
            documents: List of document dictionaries to ingest
            
        Returns:
            Dict with ingestion results and statistics
        """
        if not self.progress or not self.progress.documents:
            raise ValueError("No ingestion session started. Call start_ingestion() first.")
            
        # Create document lookup
        doc_lookup = {self._generate_doc_id(doc): doc for doc in documents}
        
        # Process pending documents
        for doc_id, checkpoint in self.progress.documents.items():
            if checkpoint.status != DocumentStatus.PENDING:
                continue
                
            if doc_id not in doc_lookup:
                logger.warning(f"Document {checkpoint.title} not found in current batch, skipping")
                checkpoint.status = DocumentStatus.SKIPPED
                self.progress.skipped_documents += 1
                continue
            
            doc = doc_lookup[doc_id]
            self._process_single_document(doc, checkpoint)
            
            # Update progress
            self.progress.last_updated = datetime.now()
            self._save_progress()
            
            # Call progress callback
            if self.on_progress_update:
                self.on_progress_update(self.get_progress_summary())

        # Final save
        self._save_progress()
        
        return self.get_progress_summary()

    def _process_single_document(self, doc: Dict, checkpoint: DocumentCheckpoint):
        """Process a single document with error handling"""
        if not self.progress:
            return
            
        # Ensure RAG service is initialized
        if not self.rag_service:
            self.rag_service = EnhancedRAGService()
            
        try:
            # Call start callback
            if self.on_document_start:
                self.on_document_start(checkpoint)
            
            logger.info(f"Processing document: {checkpoint.title}")
            checkpoint.status = DocumentStatus.PROCESSING
            self.progress.current_document = checkpoint.title
            
            # Process document through RAG service
            self.rag_service.add_documents([doc])
            
            # Mark as completed
            checkpoint.status = DocumentStatus.COMPLETED
            checkpoint.processed_at = datetime.now()
            checkpoint.error_message = None
            self.progress.completed_documents += 1
            self.progress.current_document = None
            
            # Count chunks (estimate)
            chunk_count = len(doc['content']) // 1000  # Rough estimate
            checkpoint.chunk_count = chunk_count
            
            logger.info(f"✓ Completed: {checkpoint.title} (~{chunk_count} chunks)")
            
            # Call completion callback
            if self.on_document_complete:
                self.on_document_complete(checkpoint)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"✗ Failed to process {checkpoint.title}: {error_msg}")
            
            checkpoint.retry_count += 1
            checkpoint.error_message = error_msg
            
            if checkpoint.retry_count >= self.max_retries:
                checkpoint.status = DocumentStatus.FAILED
                self.progress.failed_documents += 1
                logger.error(f"Giving up on {checkpoint.title} after {self.max_retries} retries")
                
                # Call failure callback
                if self.on_document_failed:
                    self.on_document_failed(checkpoint)
            else:
                checkpoint.status = DocumentStatus.PENDING
                logger.info(f"Will retry {checkpoint.title} (attempt {checkpoint.retry_count + 1}/{self.max_retries})")
            
            self.progress.current_document = None

    def get_progress_summary(self) -> Dict:
        """Get current progress summary"""
        if not self.progress or not self.progress.documents:
            return {}
            
        pending_count = sum(1 for c in self.progress.documents.values() if c.status == DocumentStatus.PENDING)
        processing_count = sum(1 for c in self.progress.documents.values() if c.status == DocumentStatus.PROCESSING)
        
        return {
            'session_id': self.progress.session_id,
            'started_at': self.progress.started_at.isoformat(),
            'last_updated': self.progress.last_updated.isoformat(),
            'total_documents': self.progress.total_documents,
            'completed_documents': self.progress.completed_documents,
            'failed_documents': self.progress.failed_documents,
            'skipped_documents': self.progress.skipped_documents,
            'pending_documents': pending_count,
            'processing_documents': processing_count,
            'current_document': self.progress.current_document,
            'completion_percentage': (self.progress.completed_documents / self.progress.total_documents * 100) if self.progress.total_documents > 0 else 0,
            'is_complete': (self.progress.completed_documents + self.progress.failed_documents + self.progress.skipped_documents) >= self.progress.total_documents
        }

    def get_failed_documents(self) -> List[DocumentCheckpoint]:
        """Get list of failed documents"""
        if not self.progress or not self.progress.documents:
            return []
        return [c for c in self.progress.documents.values() if c.status == DocumentStatus.FAILED]

    def retry_failed_documents(self) -> int:
        """Reset failed documents to pending for retry"""
        if not self.progress or not self.progress.documents:
            return 0
            
        retry_count = 0
        for checkpoint in self.progress.documents.values():
            if checkpoint.status == DocumentStatus.FAILED:
                checkpoint.status = DocumentStatus.PENDING
                checkpoint.retry_count = 0
                checkpoint.error_message = None
                retry_count += 1
                
        self.progress.failed_documents = 0
        self._save_progress()
        
        logger.info(f"Reset {retry_count} failed documents for retry")
        return retry_count

    def cleanup_session(self, keep_logs: bool = True):
        """Clean up session files"""
        try:
            if not keep_logs and self.progress_file.exists():
                self.progress_file.unlink()
                logger.info("Cleaned up session files")
        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

    def list_sessions(self) -> List[Dict]:
        """List all available ingestion sessions"""
        sessions = []
        try:
            for file_path in self.checkpoint_dir.glob("*_progress.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    sessions.append({
                        'session_id': data.get('session_id'),
                        'started_at': data.get('started_at'),
                        'total_documents': data.get('total_documents', 0),
                        'completed_documents': data.get('completed_documents', 0),
                        'file_path': str(file_path)
                    })
                except Exception as e:
                    logger.warning(f"Could not read session file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            
        return sessions 