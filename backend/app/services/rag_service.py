import os
import logging
from typing import List, Dict, Optional, AsyncGenerator, Any
from datetime import datetime
import asyncio
import json

# Disable ChromaDB telemetry before any ChromaDB imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_DISABLE_TELEMETRY"] = "1"

from langchain.text_splitter import RecursiveCharacterTextSplitter
# OpenAI imports (commented out for now)
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Ollama imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# New imports for enhanced features
from rank_bm25 import BM25Okapi
import numpy as np
import cohere
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.schemas import QuestionRequest, ChatResponse, SourceDocument, StreamingChunk, HybridSearchRequest

# Add this import at the top
from app.services.hierarchy_service import HierarchicalRegulatoryService

logger = logging.getLogger(__name__)

class StreamingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses"""
    
    def __init__(self):
        self.tokens = []
        self.current_chunk = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated"""
        self.tokens.append(token)
        self.current_chunk += token
        
    def get_current_chunk(self) -> str:
        """Get the current accumulated chunk"""
        chunk = self.current_chunk
        self.current_chunk = ""
        return chunk

class EnhancedRAGService:
    def __init__(self):
        # OpenAI configuration (commented out)
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # if not self.openai_api_key:
        #     raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.chroma_db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        
        # Ollama configuration
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama2")  # Default to llama2
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        
        # Cohere configuration for reranking
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if self.cohere_api_key:
            self.cohere_client = cohere.Client(self.cohere_api_key)
            logger.info("Cohere reranking enabled")
        else:
            self.cohere_client = None
            logger.warning("COHERE_API_KEY not set - reranking will be disabled")
        
        # Initialize embeddings and LLM with Ollama
        logger.info(f"Initializing Ollama with model: {self.ollama_model}")
        logger.info(f"Ollama base URL: {self.ollama_base_url}")
        
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=self.ollama_base_url,
                model=self.ollama_embedding_model
            )
            
            self.llm = OllamaLLM(
                base_url=self.ollama_base_url,
                model=self.ollama_model,
                temperature=0.1
            )
            
            # Streaming LLM for streaming responses
            self.streaming_llm = OllamaLLM(
                base_url=self.ollama_base_url,
                model=self.ollama_model,
                temperature=0.1
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama. Make sure Ollama is running: {e}")
            raise ValueError(f"Ollama initialization failed. Please ensure Ollama is installed and running at {self.ollama_base_url}")
        
        # OpenAI initialization (commented out)
        # self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        # self.llm = ChatOpenAI(
        #     model_name="gpt-3.5-turbo",
        #     temperature=0.1,
        #     openai_api_key=self.openai_api_key
        # )
        
        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory=self.chroma_db_path,
            embedding_function=self.embeddings
        )
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize BM25 for keyword search
        self.bm25_corpus = []
        self.bm25_documents = []
        self.bm25 = None
        self._initialize_bm25()
        
        # Initialize hierarchy service for regulatory precedence
        self.hierarchy_service = HierarchicalRegulatoryService()
        
        # Custom prompt template for city regulations
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert assistant for Bay Area city land development regulations. 
            Use the following pieces of context to answer the question about city regulations, zoning, parking requirements, street standards, or other development guidelines.

            Context:
            {context}

            Question: {question}

            Instructions:
            1. Provide a clear, specific answer based on the regulation context provided
            2. If the answer involves specific measurements, ratios, or requirements, be precise
            3. Always mention which city/cities the regulation applies to
            4. If the context doesn't contain enough information to fully answer the question, say so clearly
            5. Include relevant regulation section numbers or codes when available
            6. If there are variations by zone or district, mention the key differences
            7. When citing specific information, reference the source document

            Answer:"""
        )
        
        # Setup retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )

    def _initialize_bm25(self):
        """Initialize BM25 index from existing documents"""
        try:
            collection = self.vector_store._collection
            results = collection.get(include=["documents", "metadatas"])
            
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            
            if documents:
                # Improved tokenization for BM25 - lowercase and better splitting
                tokenized_corpus = []
                for doc in documents:
                    # Lowercase, split on whitespace and punctuation, remove empty tokens
                    import re
                    tokens = re.findall(r'\b\w+\b', doc.lower())
                    tokenized_corpus.append(tokens)
                
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.bm25_corpus = documents
                self.bm25_documents = [
                    {"content": doc, "metadata": meta} 
                    for doc, meta in zip(documents, metadatas)
                ]
                logger.info(f"BM25 index initialized with {len(documents)} documents")
            else:
                logger.warning("No documents found for BM25 initialization")
                
        except Exception as e:
            logger.error(f"Error initializing BM25: {str(e)}")
            self.bm25 = None

    def _keyword_search(self, query: str, k: int = 10, city: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """Perform keyword search using BM25"""
        if not self.bm25:
            return []
            
        try:
            # Improved tokenization for query - match the corpus tokenization
            import re
            tokenized_query = re.findall(r'\b\w+\b', query.lower())
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k documents with scores
            top_indices = np.argsort(scores)[::-1][:k * 2]  # Get more than needed for filtering
            
            results = []
            for idx in top_indices:
                if idx < len(self.bm25_documents):
                    doc = self.bm25_documents[idx]
                    metadata = doc["metadata"]
                    
                    # Apply filters
                    if city and metadata.get("city", "").lower() != city.lower():
                        continue
                    if category and metadata.get("category", "").lower() != category.lower():
                        continue
                        
                    results.append({
                        "content": doc["content"],
                        "metadata": metadata,
                        "score": float(scores[idx])
                    })
                    
                    if len(results) >= k:
                        break
                        
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []

    def _vector_search(self, query: str, k: int = 10, city: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """Perform vector similarity search"""
        try:
            # Build filter dict with proper ChromaDB syntax
            filter_dict = None
            if city and category:
                filter_dict = {
                    "$and": [
                        {"city": {"$eq": city}},
                        {"category": {"$eq": category}}
                    ]
                }
            elif city:
                filter_dict = {"city": {"$eq": city}}
            elif category:
                filter_dict = {"category": {"$eq": category}}
                
            # Perform similarity search
            if filter_dict:
                docs = self.vector_store.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(1 - score)  # Convert distance to similarity
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []

    def _hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7, 
                      city: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
        """Perform hybrid search combining vector and keyword search"""
        try:
            # Get results from both search methods
            vector_results = self._vector_search(query, k=k*2, city=city, category=category)
            keyword_results = self._keyword_search(query, k=k*2, city=city, category=category)
            
            # Normalize scores to [0, 1]
            if vector_results:
                max_vector_score = max(r["score"] for r in vector_results)
                if max_vector_score > 0:
                    for r in vector_results:
                        r["score"] = r["score"] / max_vector_score
                        
            if keyword_results:
                max_keyword_score = max(r["score"] for r in keyword_results)
                if max_keyword_score > 0:
                    for r in keyword_results:
                        r["score"] = r["score"] / max_keyword_score
            
            # Combine results
            combined_docs = {}
            
            # Add vector results
            for result in vector_results:
                content = result["content"]
                combined_docs[content] = {
                    "content": content,
                    "metadata": result["metadata"],
                    "vector_score": result["score"],
                    "keyword_score": 0.0
                }
            
            # Add keyword results
            for result in keyword_results:
                content = result["content"]
                if content in combined_docs:
                    combined_docs[content]["keyword_score"] = result["score"]
                else:
                    combined_docs[content] = {
                        "content": content,
                        "metadata": result["metadata"],
                        "vector_score": 0.0,
                        "keyword_score": result["score"]
                    }
            
            # Calculate hybrid scores
            final_results = []
            for doc_data in combined_docs.values():
                hybrid_score = (alpha * doc_data["vector_score"] + 
                              (1 - alpha) * doc_data["keyword_score"])
                final_results.append({
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "score": hybrid_score
                })
            
            # Sort by hybrid score and return top-k
            final_results.sort(key=lambda x: x["score"], reverse=True)
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            # Fallback to vector search
            return self._vector_search(query, k=k, city=city, category=category)

    def _rerank_documents(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using Cohere"""
        if not self.cohere_client or len(documents) <= top_k:
            return documents[:top_k]
            
        try:
            # Prepare documents for reranking
            doc_texts = [doc["content"] for doc in documents]
            
            # Use Cohere rerank
            reranked = self.cohere_client.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=doc_texts,
                top_k=top_k
            )
            
            # Reorder original documents based on reranking
            reranked_docs = []
            for result in reranked.results:
                original_doc = documents[result.index]
                original_doc["score"] = float(result.relevance_score)
                reranked_docs.append(original_doc)
                
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return documents[:top_k]

    def add_documents(self, documents: List[Dict]) -> None:
        """Add documents to the vector store and update BM25 index"""
        try:
            langchain_docs = []
            
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc["content"])
                
                for i, chunk in enumerate(chunks):
                    metadata = {
                        "title": doc["title"],
                        "city": doc["city"],
                        "category": doc["category"],
                        "chunk_index": i,
                        "source": doc.get("source", ""),
                        "source_url": doc.get("source_url", ""),
                        **doc.get("metadata", {})
                    }
                    
                    langchain_docs.append(Document(
                        page_content=chunk,
                        metadata=metadata
                    ))
            
            # Add to vector store
            logger.info(f"Creating embeddings with Ollama ({self.ollama_embedding_model})...")
            self.vector_store.add_documents(langchain_docs)
            self.vector_store.persist()
            
            # Reinitialize BM25 with new documents
            self._initialize_bm25()
            
            logger.info(f"Added {len(langchain_docs)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    async def query_streaming(self, request: QuestionRequest) -> AsyncGenerator[StreamingChunk, None]:
        """Query the RAG system with streaming response"""
        try:
            # Get relevant documents
            documents = self._search_documents(request)
            
            # Enhanced "not included" detection BEFORE LLM call
            should_skip_llm = False
            
            # Check 1: No documents found
            if not documents:
                should_skip_llm = True
            
            # Check 2: Low relevance scores
            elif max(doc.get("score", 0) for doc in documents) < 0.5:
                should_skip_llm = True
            
            # Check 3: Question is about non-regulation topics
            question_lower = request.question.lower()
            non_regulation_topics = [
                "tax", "taxes", "taxation", "business tax", "income tax",
                "license", "licensing", "permits", "fees", "business license",
                "insurance", "liability", "employment", "hiring", "payroll",
                "sales tax", "property tax", "revenue", "gross sales"
            ]
            
            if any(topic in question_lower for topic in non_regulation_topics):
                should_skip_llm = True
            
            # If we should skip LLM, return "not included" immediately
            if should_skip_llm:
                not_included_response = self._generate_not_included_response(
                    request.question, request.city, request.category
                )
                yield StreamingChunk(chunk=not_included_response, is_final=True, sources=[])
                return
            
            # Format context for LLM
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('title', 'Unknown')} ({doc['metadata'].get('city', 'Unknown')})\n{doc['content']}"
                for doc in documents
            ])
            
            # Create prompt
            prompt = self.prompt_template.format(context=context, question=request.question)
            
            # Generate response with proper streaming
            try:
                import asyncio
                from langchain.callbacks import AsyncCallbackHandler
                
                class AsyncStreamingCallbackHandler(AsyncCallbackHandler):
                    def __init__(self):
                        self.tokens = []
                        self.current_chunk = ""
                        self.chunk_queue = asyncio.Queue()
                        
                    async def on_llm_new_token(self, token: str, **kwargs) -> None:
                        """Called when a new token is generated"""
                        self.tokens.append(token)
                        self.current_chunk += token
                        
                        # Send chunk every few tokens or when we hit word boundaries
                        if len(self.current_chunk) >= 10 or token.endswith((' ', '\n', '.', '!', '?')):
                            await self.chunk_queue.put(self.current_chunk)
                            self.current_chunk = ""
                    
                    async def on_llm_end(self, response, **kwargs) -> None:
                        """Called when LLM finishes"""
                        # Send any remaining content
                        if self.current_chunk:
                            await self.chunk_queue.put(self.current_chunk)
                        # Signal end
                        await self.chunk_queue.put(None)
                
                # Create callback handler
                callback_handler = AsyncStreamingCallbackHandler()
                
                # Create streaming LLM with callback
                streaming_llm = OllamaLLM(
                    base_url=self.ollama_base_url,
                    model=self.ollama_model,
                    temperature=0.1,
                    callbacks=[callback_handler]
                )
                
                # Run LLM in background task
                async def run_llm():
                    try:
                        await streaming_llm.ainvoke(prompt)
                    except Exception as e:
                        logger.error(f"LLM streaming error: {e}")
                        await callback_handler.chunk_queue.put(None)
                
                # Start LLM task
                llm_task = asyncio.create_task(run_llm())
                
                # Stream chunks as they arrive
                full_response = ""
                while True:
                    try:
                        # Wait for next chunk with timeout
                        chunk = await asyncio.wait_for(
                            callback_handler.chunk_queue.get(), 
                            timeout=30.0
                        )
                        
                        if chunk is None:  # End of stream
                            break
                            
                        full_response += chunk
                        yield StreamingChunk(chunk=chunk, is_final=False)
                        
                    except asyncio.TimeoutError:
                        logger.warning("Streaming timeout")
                        break
                
                # Wait for LLM task to complete
                await llm_task
                
                # Final relevance check on the response
                if full_response and not self._check_answer_relevance(documents, request.question, full_response):
                    not_included_response = self._generate_not_included_response(
                        request.question, request.city, request.category
                    )
                    yield StreamingChunk(chunk=not_included_response, is_final=True, sources=[])
                    return
                
                # Send final chunk with sources
                sources = self._format_source_documents(documents)
                yield StreamingChunk(chunk="", is_final=True, sources=sources)
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(f"LLM error: {error_msg}")
                # Return "not included" on any LLM error
                error_response = self._generate_not_included_response(
                    request.question, request.city, request.category
                ) + "\n\n(Technical error occurred while processing your question.)"
                yield StreamingChunk(chunk=error_response, is_final=True, sources=[])
            
        except Exception as e:
            logger.error(f"Error in streaming query: {str(e)}")
            yield StreamingChunk(
                chunk=f"I couldn't find information about your question in the Bay Area regulation documents. Please try rephrasing your question or ask about zoning or street standards.",
                is_final=True
            )

    def _search_documents(self, request: QuestionRequest) -> List[Dict]:
        """Search documents based on request parameters"""
        search_type = request.search_type or "hybrid"
        k = request.context_limit or 5
        
        # Perform search based on type
        if search_type == "vector":
            documents = self._vector_search(
                request.question, k=k*2, 
                city=request.city, category=request.category
            )
        elif search_type == "keyword":
            documents = self._keyword_search(
                request.question, k=k*2,
                city=request.city, category=request.category
            )
        else:  # hybrid
            documents = self._hybrid_search(
                request.question, k=k*2, alpha=0.7,
                city=request.city, category=request.category
            )
        
        # Apply reranking if requested and available
        if request.use_reranking and self.cohere_client:
            documents = self._rerank_documents(request.question, documents, top_k=k)
        else:
            documents = documents[:k]
            
        return documents

    def _format_source_documents(self, documents: List[Dict]) -> List[SourceDocument]:
        """Format documents as SourceDocument objects"""
        sources = []
        for doc in documents:
            metadata = doc["metadata"]
            sources.append(SourceDocument(
                content=doc["content"],
                metadata=metadata,
                score=doc.get("score", 1.0),
                title=metadata.get("title"),
                city=metadata.get("city"),
                category=metadata.get("category"),
                chunk_index=metadata.get("chunk_index"),
                source_url=metadata.get("source_url")
            ))
        return sources

    def _check_answer_relevance(self, documents: List[Dict], question: str, answer: str) -> bool:
        """Check if the answer is relevant based on document scores and content"""
        # Check 1: No documents found
        if not documents:
            return False
        
        # Check 2: Low relevance scores (all documents have very low scores)
        max_score = max(doc.get("score", 0) for doc in documents)
        if max_score < 0.5:  # Increased threshold for relevance
            return False
        
        # Check 3: Question-answer topic mismatch detection
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # If question is about topics not in our documents, flag it
        non_regulation_topics = [
            "tax", "taxes", "taxation", "business tax", "income tax",
            "license", "licensing", "permits", "fees", "business license",
            "insurance", "liability", "employment", "hiring", "payroll",
            "sales tax", "property tax", "revenue", "gross sales",
            "accounting", "financial", "banking", "loans"
        ]
        
        question_about_non_regulation = any(topic in question_lower for topic in non_regulation_topics)
        answer_mentions_non_regulation = any(topic in answer_lower for topic in non_regulation_topics)
        
        # If question asks about non-regulation topics and answer mentions them, likely hallucination
        if question_about_non_regulation and answer_mentions_non_regulation:
            return False
        
        # Check 4: Answer indicates insufficient information
        insufficient_indicators = [
            "don't have enough information",
            "context doesn't contain",
            "not provided in the context",
            "cannot answer",
            "insufficient information",
            "not mentioned",
            "not specified",
            "i don't have",
            "the provided context does not",
            "no information about"
        ]
        
        for indicator in insufficient_indicators:
            if indicator in answer_lower:
                return False
        
        return True

    def _generate_not_included_response(self, question: str, city: Optional[str] = None, category: Optional[str] = None) -> str:
        """Generate a helpful 'not included' response"""
        base_message = "I couldn't find information about your question in the Bay Area regulation documents."
        
        # Add context about what's available
        available_info = []
        if city:
            available_info.append(f"documents for {city}")
        else:
            available_info.append("documents for San Francisco and Sunnyvale")
            
        if category:
            if category == "zoning":
                available_info.append("zoning regulations")
            elif category == "street_standards":
                available_info.append("street standards")
        else:
            available_info.append("zoning regulations and street standards")
        
        suggestion = f"\n\nThe available documents cover {' and '.join(available_info)}. You might want to:\n" \
                    f"• Rephrase your question to focus on zoning or street standards\n" \
                    f"• Ask about San Francisco or Sunnyvale specifically\n" \
                    f"• Try using different keywords related to land development regulations"
        
        return base_message + suggestion

    def query(self, request: QuestionRequest) -> ChatResponse:
        """Query the RAG system with a question"""
        try:
            # Get relevant documents
            documents = self._search_documents(request)
            
            # Format context
            context = "\n\n".join([
                f"Source: {doc['metadata'].get('title', 'Unknown')} ({doc['metadata'].get('city', 'Unknown')})\n{doc['content']}"
                for doc in documents
            ])
            
            # Generate response
            prompt = self.prompt_template.format(context=context, question=request.question)
            logger.info(f"Processing question with Ollama model: {self.ollama_model}")
            
            response = self.llm.invoke(prompt)
            
            # Format source documents
            sources = self._format_source_documents(documents)
            
            # Check answer relevance
            if not self._check_answer_relevance(documents, request.question, response):
                response = self._generate_not_included_response(request.question, request.city, request.category)
            
            return ChatResponse(
                answer=response,
                sources=sources,
                query=request.question,
                city_filter=request.city,
                category_filter=request.category,
                search_type=request.search_type or "hybrid",
                reranking_used=request.use_reranking and self.cohere_client is not None,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error querying RAG system: {str(e)}")
            raise

    def hybrid_search(self, request: HybridSearchRequest) -> List[SourceDocument]:
        """Perform hybrid search and return formatted results"""
        try:
            documents = self._hybrid_search(
                request.query,
                k=request.limit or 10,
                alpha=request.alpha or 0.7,
                city=request.city,
                category=request.category
            )
            
            return self._format_source_documents(documents)
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def get_similar_documents(self, query: str, k: int = 5, city: Optional[str] = None, 
                            category: Optional[str] = None) -> List[SourceDocument]:
        """Get similar documents without generating an answer"""
        try:
            documents = self._vector_search(query, k=k, city=city, category=category)
            return self._format_source_documents(documents)
            
        except Exception as e:
            logger.error(f"Error retrieving similar documents: {str(e)}")
            raise

    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database"""
        try:
            # Get collection info
            collection = self.vector_store._collection
            total_chunks = collection.count()
            
            # Get unique cities and categories, and count unique documents
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            cities = set()
            categories = set()
            unique_documents = set()
            
            for metadata in metadatas:
                if metadata.get("city"):
                    cities.add(metadata["city"])
                if metadata.get("category"):
                    categories.add(metadata["category"])
                # Count unique documents by title
                if metadata.get("title"):
                    unique_documents.add(metadata["title"])
            
            return {
                "total_documents": len(unique_documents),  # Count unique documents, not chunks
                "total_chunks": total_chunks,  # Also provide chunk count for reference
                "cities": list(cities),
                "categories": list(categories),
                "embedding_model": self.ollama_embedding_model,
                "llm_model": self.ollama_model,
                "hybrid_search_enabled": self.bm25 is not None,
                "reranking_enabled": self.cohere_client is not None,
                "bm25_documents": len(self.bm25_corpus) if self.bm25_corpus else 0,
                "features": {
                    "vector_search": True,
                    "keyword_search": self.bm25 is not None,
                    "hybrid_search": self.bm25 is not None,
                    "reranking": self.cohere_client is not None,
                    "streaming": True,
                    "category_filtering": True,
                    "hierarchy": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}

    def query_with_hierarchy(self, request: QuestionRequest) -> Dict[str, Any]:
        """Query with regulatory hierarchy logic"""
        try:
            # 1. Get documents from all jurisdiction levels
            documents = self._search_documents(request)
            
            # 2. Enhance documents with jurisdiction metadata and convert to SourceDocument objects
            enhanced_documents = []
            for doc in documents:
                source_doc = SourceDocument(
                    content=doc['content'],
                    metadata=doc['metadata'],
                    score=doc.get('score', 1.0),
                    title=doc['metadata'].get('title'),
                    city=doc['metadata'].get('city'),
                    category=doc['metadata'].get('category'),
                    chunk_index=doc['metadata'].get('chunk_index'),
                    source_url=doc['metadata'].get('source_url'),
                    jurisdiction_level=self._infer_jurisdiction_level(doc['metadata']),
                    jurisdiction_name=self._infer_jurisdiction_name(doc['metadata']),
                    regulation_type=doc['metadata'].get('category', 'general')
                )
                enhanced_documents.append(source_doc)
            
            # 3. Apply hierarchical organization and conflict detection
            hierarchy_groups = self.hierarchy_service.organize_by_hierarchy(enhanced_documents, request.question)
            hierarchy_result = self.hierarchy_service.generate_hierarchical_response(hierarchy_groups, request.question)
            
            # 4. Generate LLM response with hierarchy context
            controlling_regs = hierarchy_result['controlling_regulations']
            supporting_regs = hierarchy_result['supporting_regulations']
            
            # Format context with hierarchy awareness
            if controlling_regs:
                context_parts = []
                context_parts.append("=== CONTROLLING REGULATIONS ===")
                for doc in controlling_regs:
                    context_parts.append(f"Source: {doc.metadata.get('title', 'Unknown')} ({doc.jurisdiction_name or doc.metadata.get('city', 'Unknown')})")
                    context_parts.append(doc.content)
                
                if supporting_regs:
                    context_parts.append("\n=== SUPPORTING/REFERENCE REGULATIONS ===")
                    for doc in supporting_regs[:3]:  # Limit supporting docs
                        context_parts.append(f"Source: {doc.metadata.get('title', 'Unknown')} ({doc.jurisdiction_name or doc.metadata.get('city', 'Unknown')})")
                        context_parts.append(doc.content)
                
                context = "\n\n".join(context_parts)
            else:
                # Fallback to regular context if no hierarchy detected
                context = "\n\n".join([
                    f"Source: {doc.metadata.get('title', 'Unknown')} ({doc.metadata.get('city', 'Unknown')})\n{doc.content}"
                    for doc in documents
                ])
            
            # Enhanced prompt for hierarchical context
            hierarchy_prompt = f"""You are an expert assistant for Bay Area city land development regulations with knowledge of regulatory hierarchy.

When multiple jurisdictions have different requirements, apply this hierarchy (most specific wins):
1. Special Districts/Master Plans (highest authority)
2. City/Municipal regulations
3. County regulations  
4. State regulations (lowest authority)

Context:
{context}

Question: {request.question}

Instructions:
1. If there are conflicting regulations from different jurisdictions, clearly state which one controls and why
2. Provide a clear, specific answer based on the controlling regulation
3. Mention all applicable jurisdictions but emphasize the controlling authority
4. Include specific measurements, ratios, or requirements from the controlling regulation
5. Always mention which jurisdiction's regulation applies
6. If regulations don't conflict, provide a comprehensive answer covering all applicable rules

Answer:"""
            
            # Generate response
            logger.info(f"Processing hierarchical question with Ollama model: {self.ollama_model}")
            response = self.llm.invoke(hierarchy_prompt)
            
            # Check answer relevance
            if not self._check_answer_relevance(documents, request.question, response):
                response = self._generate_not_included_response(request.question, request.city, request.category)
            
            # Combine all sources for response
            all_sources = controlling_regs + supporting_regs
            
            return {
                "answer": response,
                "sources": all_sources,
                "hierarchy_explanation": hierarchy_result['hierarchy_explanation'],
                "controlling_regulations": controlling_regs,
                "supporting_regulations": supporting_regs,
                "conflicts_detected": hierarchy_result['conflicts_detected'],
                "query": request.question,
                "city_filter": request.city,
                "category_filter": request.category,
                "search_type": request.search_type or "hybrid",
                "reranking_used": request.use_reranking and self.cohere_client is not None,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in hierarchical query: {str(e)}")
            # Fallback to regular query if hierarchy fails
            return self.query(request).__dict__
    
    def _infer_jurisdiction_level(self, metadata: Dict) -> str:
        """Infer jurisdiction level from document metadata"""
        title = metadata.get('title', '').lower()
        city = metadata.get('city', '').lower()
        category = metadata.get('category', '').lower()
        
        # Check for district/master plan indicators
        if any(keyword in title for keyword in ['district', 'master plan', 'specific plan', 'overlay']):
            return 'district'
        
        # Check for city-level indicators
        if city in ['sunnyvale', 'san francisco'] or 'municipal' in title or 'city' in title:
            return 'city'
        
        # Check for county indicators
        if 'county' in title or 'santa clara' in title:
            return 'county'
        
        # Check for state indicators
        if 'california' in title or 'state' in title or 'cbc' in title:
            return 'state'
        
        # Default to city level
        return 'city'
    
    def _infer_jurisdiction_name(self, metadata: Dict) -> str:
        """Infer jurisdiction name from document metadata"""
        title = metadata.get('title', '').lower()
        city = metadata.get('city', '')
        
        # Check for specific district names
        if 'downtown' in title:
            return f'Downtown District - {city}'
        elif 'transit' in title:
            return f'Transit District - {city}'
        elif 'district' in title or 'master plan' in title:
            return f'Special District - {city}'
        
        # Return city name or default
        return city or 'Unknown Jurisdiction'

# Backward compatibility
RAGService = EnhancedRAGService 