import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB-based vector store for document embeddings"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.client = None
        self.collection = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure the vector DB directory exists
            Config.ensure_directories()
            
            # Initialize ChromaDB with persistent storage
            self.client = chromadb.PersistentClient(
                path=Config.VECTOR_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized vector store with {self.collection.count()} existing documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to the vector store"""
        if not chunks:
            logger.warning("No chunks provided to add")
            return False
        
        try:
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                content = chunk["content"]
                metadata = chunk["metadata"]
                
                # Create unique ID based on content hash and source
                chunk_id = f"{metadata['source']}_{metadata['chunk_id']}_{metadata['content_hash'][:8]}"
                
                documents.append(content)
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            # Check for existing documents to avoid duplicates
            existing_ids = set()
            try:
                existing_data = self.collection.get(ids=ids)
                if existing_data and existing_data['ids']:
                    existing_ids = set(existing_data['ids'])
            except Exception as e:
                logger.warning(f"Could not check for existing documents: {str(e)}")
            
            # Filter out existing documents
            new_documents = []
            new_metadatas = []
            new_ids = []
            
            for doc, meta, doc_id in zip(documents, metadatas, ids):
                if doc_id not in existing_ids:
                    new_documents.append(doc)
                    new_metadatas.append(meta)
                    new_ids.append(doc_id)
            
            if not new_documents:
                logger.info("All documents already exist in the vector store")
                return True
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(new_documents)} new chunks")
            embeddings = self.embedding_model.encode(new_documents).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=new_documents,
                metadatas=new_metadatas,
                ids=new_ids,
                embeddings=embeddings
            )
            
            logger.info(f"Successfully added {len(new_documents)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            return False
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity_score = 1 - distance  # Convert distance to similarity
                    
                    if similarity_score >= Config.SIMILARITY_THRESHOLD:
                        search_results.append({
                            "content": doc,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "rank": i + 1
                        })
            
            logger.info(f"Semantic search returned {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return []
    
    def keyword_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        try:
            # Get all documents from ChromaDB
            all_docs = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_docs['documents']:
                return []
            
            # Simple keyword matching with scoring
            query_terms = query.lower().split()
            scored_results = []
            
            for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
                doc_lower = doc.lower()
                score = 0
                
                # Count keyword matches
                for term in query_terms:
                    score += doc_lower.count(term)
                
                # Boost score for exact phrase matches
                if query.lower() in doc_lower:
                    score += len(query_terms) * 2
                
                if score > 0:
                    scored_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "keyword_score": score,
                        "normalized_score": min(score / len(query_terms), 1.0)
                    })
            
            # Sort by score and take top_k
            scored_results.sort(key=lambda x: x["keyword_score"], reverse=True)
            keyword_results = scored_results[:top_k]
            
            # Add rank
            for i, result in enumerate(keyword_results):
                result["rank"] = i + 1
            
            logger.info(f"Keyword search returned {len(keyword_results)} results for query: {query[:50]}...")
            return keyword_results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search with weighted scoring"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        # Get results from both search methods
        semantic_results = self.semantic_search(query, top_k * 2)  # Get more for better fusion
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine and re-rank results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result["metadata"]["content_hash"]
            combined_results[doc_id] = {
                "content": result["content"],
                "metadata": result["metadata"],
                "semantic_score": result["similarity_score"],
                "keyword_score": 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result["metadata"]["content_hash"]
            if doc_id in combined_results:
                combined_results[doc_id]["keyword_score"] = result["normalized_score"]
            else:
                combined_results[doc_id] = {
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "semantic_score": 0.0,
                    "keyword_score": result["normalized_score"]
                }
        
        # Calculate hybrid scores
        final_results = []
        for doc_id, result in combined_results.items():
            hybrid_score = (
                Config.SEMANTIC_WEIGHT * result["semantic_score"] + 
                Config.KEYWORD_WEIGHT * result["keyword_score"]
            )
            
            if hybrid_score > 0:
                final_results.append({
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "semantic_score": result["semantic_score"],
                    "keyword_score": result["keyword_score"],
                    "hybrid_score": hybrid_score
                })
        
        # Sort by hybrid score and take top_k
        final_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        final_results = final_results[:top_k]
        
        # Add ranks
        for i, result in enumerate(final_results):
            result["rank"] = i + 1
        
        logger.info(f"Hybrid search returned {len(final_results)} results for query: {query[:50]}...")
        return final_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_size = min(count, 100)
            if count > 0:
                sample_docs = self.collection.get(
                    limit=sample_size,
                    include=['metadatas']
                )
                
                # Analyze file types
                file_types = {}
                sources = set()
                
                for metadata in sample_docs['metadatas']:
                    file_type = metadata.get('file_type', 'unknown')
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                    sources.add(metadata.get('source', 'unknown'))
                
                return {
                    "total_chunks": count,
                    "unique_sources": len(sources),
                    "file_types": file_types,
                    "sample_sources": list(sources)[:10]  # Show first 10 sources
                }
            else:
                return {
                    "total_chunks": 0,
                    "unique_sources": 0,
                    "file_types": {},
                    "sample_sources": []
                }
                
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from the vector store"""
        try:
            # Delete and recreate collection
            self.client.delete_collection("documents")
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Successfully cleared vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            return False
    
    def delete_by_source(self, source_name: str) -> bool:
        """Delete all chunks from a specific source file"""
        try:
            # Get all documents from the source
            all_docs = self.collection.get(
                where={"source": source_name},
                include=['ids']
            )
            
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
                logger.info(f"Deleted {len(all_docs['ids'])} chunks from source: {source_name}")
                return True
            else:
                logger.info(f"No documents found for source: {source_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete documents from source {source_name}: {str(e)}")
            return False