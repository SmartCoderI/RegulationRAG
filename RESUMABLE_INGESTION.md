# 🔄 Resumable Document Ingestion Pipeline

Your enhanced RAG system now includes a robust resumable ingestion pipeline that can handle interruptions, track progress, and resume from where it left off.

## 🌟 Key Features

### ✅ **Checkpoint System**
- **Progress Tracking**: Saves after each document
- **File Change Detection**: Re-processes modified files
- **Session Management**: Multiple ingestion sessions
- **Error Recovery**: Automatic retry with configurable limits

### 📊 **Real-time Progress Reporting**
- **Live Updates**: See processing status in real-time
- **ETA Calculation**: Estimated time to completion
- **Detailed Logging**: Comprehensive logs saved to file
- **Statistics**: Success rates, timing, error details

### 🛡️ **Robust Error Handling**
- **Graceful Interruption**: Ctrl+C handling
- **Automatic Retry**: Failed documents retry automatically
- **Error Isolation**: One failed document doesn't stop the whole process
- **Recovery Options**: Resume from any point

## 🚀 Quick Start

### 1. **Basic Usage**

```bash
# Navigate to backend
cd backend

# Activate virtual environment
source venv/bin/activate

# Ingest sample documents
python scripts/ingest_resumable.py --source-type sample

# Ingest user documents
python scripts/ingest_resumable.py --source-type user --source-dir "/path/to/your/docs"

# Ingest PDF documents
python scripts/ingest_resumable.py --source-type pdf --source-dir "/path/to/pdfs"
```

### 2. **Resume Interrupted Session**

```bash
# If ingestion was interrupted, resume with session ID
python scripts/ingest_resumable.py --source-type pdf --source-dir "/path/to/pdfs" --session-id 20241215_143052
```

### 3. **Session Management**

```bash
# List all available sessions
python scripts/ingest_resumable.py --list-sessions

# Retry failed documents in a session
python scripts/ingest_resumable.py --source-type pdf --source-dir "/path/to/pdfs" --session-id 20241215_143052 --retry-failed

# Clean up completed session
python scripts/ingest_resumable.py --cleanup-session 20241215_143052
```

## 📋 Command Reference

### **Required Arguments**
- `--source-type {sample,user,pdf}`: Type of documents to ingest
  - `sample`: Built-in sample regulation documents
  - `user`: Documents from user directory (mixed formats)
  - `pdf`: PDF documents only

### **Optional Arguments**
- `--source-dir PATH`: Source directory for user/pdf documents
- `--session-id ID`: Resume specific session (e.g., `20241215_143052`)
- `--max-retries N`: Maximum retries per document (default: 3)
- `--checkpoint-dir PATH`: Directory for checkpoint files (default: `./checkpoints`)

### **Session Management**
- `--list-sessions`: Show all available sessions
- `--cleanup-session ID`: Remove session files
- `--retry-failed`: Reset failed documents for retry

## 🔧 Advanced Usage

### **1. Custom Retry Limits**
```bash
# Allow up to 5 retries per document
python scripts/ingest_resumable.py --source-type pdf --source-dir "/path/to/pdfs" --max-retries 5
```

### **2. Custom Checkpoint Directory**
```bash
# Use custom checkpoint location
python scripts/ingest_resumable.py --source-type pdf --source-dir "/path/to/pdfs" --checkpoint-dir "/custom/checkpoints"
```

### **3. Handle Large Document Collections**
```bash
# Start ingestion of 1000+ documents
python scripts/ingest_resumable.py --source-type pdf --source-dir "/large/collection"

# If interrupted, resume exactly where you left off
python scripts/ingest_resumable.py --source-type pdf --source-dir "/large/collection" --session-id <SESSION_ID>
```

## 📊 Progress Monitoring

### **Real-time Output**
```
2024-12-15 14:30:52 - INFO - Found 127 documents to ingest
2024-12-15 14:30:52 - INFO - 📊 Document Summary:
2024-12-15 14:30:52 - INFO -    Cities: {'San Francisco': 45, 'Sunnyvale': 32, 'General': 50}
2024-12-15 14:30:52 - INFO -    Categories: {'zoning': 67, 'street_standards': 35, 'general': 25}
2024-12-15 14:30:52 - INFO - 🚀 Starting ingestion session: 20241215_143052
2024-12-15 14:30:53 - INFO - 📄 Starting: San Francisco - Zoning Code Section 1
2024-12-15 14:30:55 - INFO - ✅ Completed: San Francisco - Zoning Code Section 1 (chunks: 15, time: 2.1s)
2024-12-15 14:30:57 - INFO - 📊 Progress: 5/127 (3.9%) - ETA: 245s
```

### **Checkpoint Files**
Progress is automatically saved to:
- `./checkpoints/ingestion_progress.json`: Current session progress
- `./ingestion.log`: Detailed processing log

### **Session Data Structure**
```json
{
  "session_id": "20241215_143052",
  "started_at": "2024-12-15T14:30:52",
  "total_documents": 127,
  "completed_documents": 45,
  "failed_documents": 2,
  "documents": {
    "doc_abc123": {
      "title": "San Francisco - Zoning Code",
      "status": "completed",
      "processed_at": "2024-12-15T14:31:15",
      "chunk_count": 15
    }
  }
}
```

## 🛠️ Integration with Enhanced RAG System

### **Automatic Features**
The resumable ingestion system automatically uses all enhanced RAG features:

- **🔮 Hybrid Search**: Documents indexed for both vector and keyword search
- **⚡ Enhanced Processing**: Uses the enhanced RAG service with all features
- **📂 Rich Metadata**: Includes source URLs for citations
- **🔗 Smart Categorization**: Automatic city and category detection

### **File Change Detection**
The system automatically detects when files have changed:

```bash
# First run - processes all documents
python scripts/ingest_resumable.py --source-type pdf --source-dir "/docs"

# Modify a PDF file in the directory
# Run again - only re-processes changed files
python scripts/ingest_resumable.py --source-type pdf --source-dir "/docs" --session-id <SESSION_ID>
```

## 🚨 Error Handling & Recovery

### **Automatic Retry**
- Failed documents automatically retry up to `max-retries` times
- Each retry includes exponential backoff
- Errors are logged with full details

### **Graceful Interruption**
Press `Ctrl+C` to stop gracefully:
```
^C2024-12-15 14:35:22 - INFO - Shutdown requested. Finishing current document...
2024-12-15 14:35:24 - INFO - ⏸️  Ingestion interrupted by user
2024-12-15 14:35:24 - INFO - 💾 Progress saved. Resume with: --session-id 20241215_143052
```

### **Recovery Options**

**1. Resume Interrupted Session**
```bash
python scripts/ingest_resumable.py --source-type pdf --source-dir "/docs" --session-id 20241215_143052
```

**2. Retry Failed Documents**
```bash
python scripts/ingest_resumable.py --source-type pdf --source-dir "/docs" --session-id 20241215_143052 --retry-failed
```

**3. Check Session Status**
```bash
python scripts/ingest_resumable.py --list-sessions
```

## 🎯 Best Practices

### **1. Large Document Collections**
- Use `--max-retries 5` for unstable networks/systems
- Monitor the `ingestion.log` file for detailed progress
- Keep checkpoint files until ingestion is complete

### **2. File Organization**
```
your-documents/
├── san_francisco_zoning.pdf
├── sunnyvale_street_standards.pdf
├── general_design_guidelines.pdf
└── historic_preservation_rules.pdf
```

### **3. Monitoring Progress**
```bash
# In one terminal - start ingestion
python scripts/ingest_resumable.py --source-type pdf --source-dir "/docs"

# In another terminal - monitor logs
tail -f ingestion.log
```

### **4. Cleanup After Success**
```bash
# List completed sessions
python scripts/ingest_resumable.py --list-sessions

# Clean up successful sessions
python scripts/ingest_resumable.py --cleanup-session 20241215_143052
```

## 📈 Performance Tips

### **1. Optimize Document Size**
- PDFs under 10MB process faster
- Split very large documents for better parallelization
- Use clear, descriptive filenames for better categorization

### **2. System Resources**
- Ensure adequate disk space for checkpoints
- Monitor memory usage with large documents
- Use SSD storage for better performance

### **3. Network Considerations**
- Ollama embeddings are computed locally (no network needed)
- Only Cohere reranking requires internet (optional)
- All processing can be done offline

## 🔍 Troubleshooting

### **Common Issues**

**1. "Session not found"**
```bash
# Check available sessions
python scripts/ingest_resumable.py --list-sessions

# Start new session if needed
python scripts/ingest_resumable.py --source-type pdf --source-dir "/docs"
```

**2. "RAG service initialization failed"**
```bash
# Ensure Ollama is running
ollama serve

# Check Ollama status
curl http://localhost:11434/api/tags
```

**3. "PDF parsing failed"**
```bash
# Install PDF dependencies
pip install PyPDF2 pypdf
```

**4. "Checkpoint directory permissions"**
```bash
# Fix permissions
chmod 755 ./checkpoints
```

### **Debug Mode**
For detailed debugging, check the ingestion log:
```bash
tail -f ingestion.log | grep ERROR
```

## 🎉 Success Metrics

After successful ingestion, you'll see:
```
2024-12-15 14:45:33 - INFO - 🎉 Ingestion completed!
2024-12-15 14:45:33 - INFO - 📊 Final Results:
2024-12-15 14:45:33 - INFO -    Session ID: 20241215_143052
2024-12-15 14:45:33 - INFO -    Total Documents: 127
2024-12-15 14:45:33 - INFO -    Completed: 125
2024-12-15 14:45:33 - INFO -    Failed: 0
2024-12-15 14:45:33 - INFO -    Skipped: 2
2024-12-15 14:45:33 - INFO -    Success Rate: 98.4%
2024-12-15 14:45:33 - INFO -    Total Time: 892.3s
```

## 🚀 Next Steps

After successful ingestion:

1. **Test Your Documents**: Use the enhanced chat interface with all new features
2. **Verify Citations**: Check that source links work properly
3. **Try Advanced Search**: Test hybrid search, filtering, and reranking
4. **Monitor Performance**: Use the enhanced health endpoint to check system status

Your documents are now ready for use with all enhanced RAG features! 🎊

---

**Need help?** Check the main `ENHANCEMENTS.md` file for the complete feature guide. 