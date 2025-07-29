import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from document_processor import DocumentProcessor
from vector_store import VectorStore
from retrieval_system import RetrievalSystem
from llm_integration import LLMIntegration
from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system orchestrating the entire pipeline"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.retrieval_system = RetrievalSystem()
        self.llm_integration = LLMIntegration()
        
        # Ensure required directories exist
        Config.ensure_directories()
        
        # Initialize session history
        self.session_history = []
        
        logger.info("RAG System initialized successfully")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the RAG system
        
        Args:
            file_path: Path to the document to ingest
        
        Returns:
            Dictionary with ingestion results and metadata
        """
        try:
            logger.info(f"Starting document ingestion: {file_path}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path
                }
            
            # Process the document
            chunks = self.document_processor.process_document(file_path)
            
            if not chunks:
                error_msg = f"No content extracted from document: {file_path}"
                logger.warning(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path,
                    "chunks_extracted": 0
                }
            
            # Add chunks to vector store
            success = self.vector_store.add_documents(chunks)
            
            if success:
                result = {
                    "success": True,
                    "file_path": file_path,
                    "filename": os.path.basename(file_path),
                    "chunks_extracted": len(chunks),
                    "file_type": chunks[0]["metadata"]["file_type"] if chunks else "unknown",
                    "ingestion_time": datetime.now().isoformat(),
                    "total_characters": sum(len(chunk["content"]) for chunk in chunks)
                }
                
                logger.info(f"Successfully ingested {len(chunks)} chunks from {file_path}")
                return result
            else:
                error_msg = f"Failed to add document chunks to vector store: {file_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path,
                    "chunks_extracted": len(chunks)
                }
                
        except Exception as e:
            error_msg = f"Document ingestion failed for {file_path}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "file_path": file_path,
                "exception": str(e)
            }
    
    def ingest_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Ingest all supported documents from a folder
        
        Args:
            folder_path: Path to the folder containing documents
        
        Returns:
            Dictionary with batch ingestion results
        """
        try:
            logger.info(f"Starting folder ingestion: {folder_path}")
            
            if not os.path.exists(folder_path):
                error_msg = f"Folder not found: {folder_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "folder_path": folder_path
                }
            
            # Process all documents in the folder
            all_chunks = self.document_processor.process_folder(folder_path)
            
            if not all_chunks:
                warning_msg = f"No content extracted from any documents in folder: {folder_path}"
                logger.warning(warning_msg)
                return {
                    "success": False,
                    "error": warning_msg,
                    "folder_path": folder_path,
                    "total_chunks": 0
                }
            
            # Group chunks by source for detailed reporting
            files_processed = {}
            for chunk in all_chunks:
                source = chunk["metadata"]["source"]
                if source not in files_processed:
                    files_processed[source] = {
                        "chunks": 0,
                        "file_type": chunk["metadata"]["file_type"]
                    }
                files_processed[source]["chunks"] += 1
            
            # Add all chunks to vector store
            success = self.vector_store.add_documents(all_chunks)
            
            if success:
                result = {
                    "success": True,
                    "folder_path": folder_path,
                    "total_files": len(files_processed),
                    "total_chunks": len(all_chunks),
                    "files_processed": files_processed,
                    "ingestion_time": datetime.now().isoformat(),
                    "total_characters": sum(len(chunk["content"]) for chunk in all_chunks)
                }
                
                logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(files_processed)} files")
                return result
            else:
                error_msg = f"Failed to add chunks to vector store from folder: {folder_path}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "folder_path": folder_path,
                    "total_chunks": len(all_chunks)
                }
                
        except Exception as e:
            error_msg = f"Folder ingestion failed for {folder_path}: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "folder_path": folder_path,
                "exception": str(e)
            }
    
    def ask_question(self, question: str, search_type: str = "hybrid", 
                    include_history: bool = True) -> Dict[str, Any]:
        """
        Ask a question and get an answer from the RAG system
        
        Args:
            question: The question to ask
            search_type: Type of search to use ("semantic", "keyword", "hybrid")
            include_history: Whether to include this Q&A in session history
        
        Returns:
            Dictionary with answer, sources, confidence, and metadata
        """
        try:
            start_time = datetime.now()
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant documents
            relevant_chunks = self.retrieval_system.retrieve(
                query=question,
                search_type=search_type,
                top_k=Config.TOP_K_RETRIEVAL
            )
            
            if not relevant_chunks:
                result = {
                    "question": question,
                    "answer": "I couldn't find any relevant information in the ingested documents to answer your question. Please make sure you have uploaded relevant documents or try rephrasing your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "retrieval_type": search_type,
                    "chunks_found": 0,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Generate answer using LLM
                answer_data = self.llm_integration.generate_answer(question, relevant_chunks)
                
                # Prepare sources information
                sources = []
                for i, chunk in enumerate(relevant_chunks):
                    source_info = {
                        "rank": i + 1,
                        "filename": chunk["metadata"]["source"],
                        "file_type": chunk["metadata"]["file_type"],
                        "chunk_id": chunk["metadata"]["chunk_id"],
                        "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                        "scores": {}
                    }
                    
                    # Add available scores
                    if "similarity_score" in chunk:
                        source_info["scores"]["similarity"] = round(chunk["similarity_score"], 3)
                    if "keyword_score" in chunk:
                        source_info["scores"]["keyword"] = round(chunk.get("keyword_score", 0), 3)
                    if "hybrid_score" in chunk:
                        source_info["scores"]["hybrid"] = round(chunk["hybrid_score"], 3)
                    
                    sources.append(source_info)
                
                result = {
                    "question": question,
                    "answer": answer_data["answer"],
                    "sources": sources,
                    "confidence": answer_data["confidence"],
                    "model_used": answer_data["model_used"],
                    "model_name": answer_data["model_name"],
                    "retrieval_type": search_type,
                    "chunks_found": len(relevant_chunks),
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add warning if confidence is low
                if "warning" in answer_data:
                    result["warning"] = answer_data["warning"]
                
                # Add token usage if available
                if "token_usage" in answer_data and answer_data["token_usage"]:
                    result["token_usage"] = answer_data["token_usage"]
            
            # Add to session history
            if include_history:
                self.session_history.append({
                    "question": question,
                    "answer": result["answer"],
                    "confidence": result["confidence"],
                    "sources_count": len(result["sources"]),
                    "timestamp": result["timestamp"]
                })
                
                # Keep only last 50 entries
                if len(self.session_history) > 50:
                    self.session_history = self.session_history[-50:]
            
            logger.info(f"Question processed successfully in {result['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process question: {str(e)}"
            logger.error(error_msg)
            return {
                "question": question,
                "answer": "I encountered an error while processing your question. Please try again or contact support.",
                "sources": [],
                "confidence": 0.0,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def search_documents(self, query: str, search_type: str = "hybrid", 
                        top_k: int = None) -> Dict[str, Any]:
        """
        Search for documents without generating an answer
        
        Args:
            query: Search query
            search_type: Type of search to use
            top_k: Number of results to return
        
        Returns:
            Dictionary with search results and metadata
        """
        try:
            if top_k is None:
                top_k = Config.TOP_K_RETRIEVAL * 2  # Return more for search
            
            logger.info(f"Searching documents for: {query[:100]}...")
            
            results = self.retrieval_system.retrieve(
                query=query,
                search_type=search_type,
                top_k=top_k
            )
            
            # Format results for return
            formatted_results = []
            for i, result in enumerate(results):
                formatted_result = {
                    "rank": i + 1,
                    "filename": result["metadata"]["source"],
                    "file_type": result["metadata"]["file_type"],
                    "content": result["content"],
                    "chunk_id": result["metadata"]["chunk_id"],
                    "scores": {}
                }
                
                # Add available scores
                if "similarity_score" in result:
                    formatted_result["scores"]["similarity"] = round(result["similarity_score"], 3)
                if "keyword_score" in result:
                    formatted_result["scores"]["keyword"] = round(result.get("keyword_score", 0), 3)
                if "hybrid_score" in result:
                    formatted_result["scores"]["hybrid"] = round(result["hybrid_score"], 3)
                
                formatted_results.append(formatted_result)
            
            return {
                "query": query,
                "search_type": search_type,
                "results": formatted_results,
                "total_results": len(formatted_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            error_msg = f"Document search failed: {str(e)}"
            logger.error(error_msg)
            return {
                "query": query,
                "search_type": search_type,
                "results": [],
                "total_results": 0,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics"""
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            
            # Get model info
            model_info = self.llm_integration.get_model_info()
            
            # Get retrieval stats
            retrieval_stats = self.retrieval_system.get_retrieval_stats()
            
            return {
                "system_status": "operational",
                "vector_store": vector_stats,
                "language_models": model_info,
                "retrieval_system": retrieval_stats,
                "session_history_length": len(self.session_history),
                "configuration": {
                    "chunk_size": Config.CHUNK_SIZE,
                    "chunk_overlap": Config.CHUNK_OVERLAP,
                    "top_k_retrieval": Config.TOP_K_RETRIEVAL,
                    "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                    "supported_extensions": Config.SUPPORTED_EXTENSIONS
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {str(e)}")
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_vector_store(self) -> Dict[str, Any]:
        """Clear all documents from the vector store"""
        try:
            success = self.vector_store.clear_collection()
            if success:
                logger.info("Vector store cleared successfully")
                return {
                    "success": True,
                    "message": "All documents have been removed from the vector store",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to clear vector store",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            error_msg = f"Failed to clear vector store: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get the current session's Q&A history"""
        return self.session_history.copy()
    
    def clear_session_history(self):
        """Clear the current session's Q&A history"""
        self.session_history.clear()
        logger.info("Session history cleared")
    
    def test_system(self) -> Dict[str, Any]:
        """Run a comprehensive system test"""
        try:
            logger.info("Running system test...")
            
            # Test retrieval system
            retrieval_test = self.retrieval_system.test_retrieval("test query")
            
            # Test vector store
            vector_stats = self.vector_store.get_collection_stats()
            
            # Test LLM integration
            model_info = self.llm_integration.get_model_info()
            
            # Overall system status
            if (retrieval_test.get("status") == "success" and 
                "error" not in vector_stats and 
                (model_info.get("openai_available") or model_info.get("models_loaded", {}).get("huggingface"))):
                system_status = "healthy"
            else:
                system_status = "degraded"
            
            return {
                "system_status": system_status,
                "retrieval_test": retrieval_test,
                "vector_store_status": "healthy" if "error" not in vector_stats else "error",
                "llm_status": "healthy" if (model_info.get("openai_available") or 
                                          model_info.get("models_loaded", {}).get("huggingface")) else "error",
                "total_documents": vector_stats.get("total_chunks", 0),
                "test_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System test failed: {str(e)}")
            return {
                "system_status": "error",
                "error": str(e),
                "test_timestamp": datetime.now().isoformat()
            }