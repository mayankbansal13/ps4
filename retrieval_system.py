import logging
from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class RetrievalSystem:
    """Handles hybrid retrieval combining semantic and keyword search"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        logger.info("Retrieval system initialized")
    
    def retrieve(self, query: str, search_type: str = "hybrid", top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on query
        
        Args:
            query: The search query
            search_type: Type of search ("semantic", "keyword", "hybrid")
            top_k: Number of results to return
        
        Returns:
            List of relevant document chunks with metadata and scores
        """
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        logger.info(f"Performing {search_type} search for query: {query[:50]}...")
        
        if search_type == "semantic":
            return self.semantic_retrieve(query, top_k)
        elif search_type == "keyword":
            return self.keyword_retrieve(query, top_k)
        elif search_type == "hybrid":
            return self.hybrid_retrieve(query, top_k)
        else:
            logger.error(f"Unknown search type: {search_type}")
            return []
    
    def semantic_retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform semantic retrieval using vector similarity"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.semantic_search(query, top_k)
            
            # Add retrieval metadata
            for result in results:
                result["retrieval_type"] = "semantic"
                result["query"] = query
            
            logger.info(f"Semantic retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Semantic retrieval failed: {str(e)}")
            return []
    
    def keyword_retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform keyword-based retrieval"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.keyword_search(query, top_k)
            
            # Add retrieval metadata
            for result in results:
                result["retrieval_type"] = "keyword"
                result["query"] = query
            
            logger.info(f"Keyword retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Keyword retrieval failed: {str(e)}")
            return []
    
    def hybrid_retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval combining semantic and keyword search"""
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        try:
            results = self.vector_store.hybrid_search(query, top_k)
            
            # Add retrieval metadata
            for result in results:
                result["retrieval_type"] = "hybrid"
                result["query"] = query
            
            logger.info(f"Hybrid retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {str(e)}")
            return []
    
    def retrieve_with_filters(self, query: str, filters: Dict[str, Any] = None, 
                            search_type: str = "hybrid", top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents with additional filtering
        
        Args:
            query: The search query
            filters: Dictionary of filters to apply (e.g., {"file_type": "pdf"})
            search_type: Type of search to perform
            top_k: Number of results to return
        
        Returns:
            Filtered list of relevant document chunks
        """
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        # Get initial results
        results = self.retrieve(query, search_type, top_k * 2)  # Get more for filtering
        
        # Apply filters if provided
        if filters and results:
            filtered_results = []
            for result in results:
                metadata = result.get("metadata", {})
                include_result = True
                
                for filter_key, filter_value in filters.items():
                    if filter_key in metadata:
                        if isinstance(filter_value, list):
                            if metadata[filter_key] not in filter_value:
                                include_result = False
                                break
                        else:
                            if metadata[filter_key] != filter_value:
                                include_result = False
                                break
                    else:
                        include_result = False
                        break
                
                if include_result:
                    filtered_results.append(result)
            
            results = filtered_results[:top_k]
            logger.info(f"Applied filters, {len(results)} results remaining")
        
        return results
    
    def multi_query_retrieve(self, queries: List[str], search_type: str = "hybrid", 
                           top_k: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries
        
        Args:
            queries: List of search queries
            search_type: Type of search to perform
            top_k: Number of results per query
        
        Returns:
            Dictionary mapping queries to their results
        """
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, search_type, top_k)
            logger.info(f"Retrieved {len(results[query])} results for query: {query[:30]}...")
        
        return results
    
    def get_similar_documents(self, document_content: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document content
        
        Args:
            document_content: The content to find similar documents for
            top_k: Number of similar documents to return
        
        Returns:
            List of similar document chunks
        """
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        # Use the document content as a query for semantic search
        results = self.semantic_retrieve(document_content, top_k)
        
        # Add metadata indicating this is a similarity search
        for result in results:
            result["retrieval_type"] = "similarity"
            result["query_type"] = "document_similarity"
        
        return results
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms
        
        Args:
            query: Original query
        
        Returns:
            List of expanded queries
        """
        # Simple query expansion (can be enhanced with more sophisticated methods)
        expanded_queries = [query]
        
        # Add variations
        words = query.lower().split()
        
        # Add query without stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        filtered_words = [word for word in words if word not in stop_words]
        if len(filtered_words) != len(words) and len(filtered_words) > 0:
            expanded_queries.append(" ".join(filtered_words))
        
        # Add individual important words as separate queries
        if len(words) > 1:
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    expanded_queries.append(word)
        
        logger.info(f"Expanded query '{query}' to {len(expanded_queries)} variations")
        return expanded_queries
    
    def retrieve_with_expansion(self, query: str, search_type: str = "hybrid", 
                              top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using query expansion
        
        Args:
            query: Original query
            search_type: Type of search to perform
            top_k: Number of results to return
        
        Returns:
            Combined and ranked results from expanded queries
        """
        if top_k is None:
            top_k = Config.TOP_K_RETRIEVAL
        
        # Expand the query
        expanded_queries = self.expand_query(query)
        
        # Retrieve for all expanded queries
        all_results = {}
        for exp_query in expanded_queries:
            results = self.retrieve(exp_query, search_type, top_k)
            for result in results:
                content_hash = result["metadata"]["content_hash"]
                if content_hash not in all_results:
                    all_results[content_hash] = result
                    all_results[content_hash]["query_matches"] = [exp_query]
                else:
                    # Boost score for multiple query matches
                    all_results[content_hash]["query_matches"].append(exp_query)
                    if "hybrid_score" in result:
                        all_results[content_hash]["hybrid_score"] = max(
                            all_results[content_hash].get("hybrid_score", 0),
                            result["hybrid_score"]
                        )
        
        # Convert back to list and sort
        final_results = list(all_results.values())
        
        # Sort by best available score
        def get_sort_key(result):
            if "hybrid_score" in result:
                return result["hybrid_score"]
            elif "similarity_score" in result:
                return result["similarity_score"]
            elif "keyword_score" in result:
                return result["keyword_score"]
            else:
                return len(result.get("query_matches", []))
        
        final_results.sort(key=get_sort_key, reverse=True)
        final_results = final_results[:top_k]
        
        # Add expanded query metadata
        for i, result in enumerate(final_results):
            result["retrieval_type"] = f"{search_type}_expanded"
            result["original_query"] = query
            result["rank"] = i + 1
        
        logger.info(f"Query expansion retrieval returned {len(final_results)} results")
        return final_results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                "vector_store_stats": vector_stats,
                "search_types_available": ["semantic", "keyword", "hybrid"],
                "features": [
                    "Multi-format document support",
                    "Hybrid search (semantic + keyword)",
                    "Query expansion",
                    "Filtered retrieval",
                    "Multi-query retrieval",
                    "Document similarity search"
                ],
                "configuration": {
                    "top_k_default": Config.TOP_K_RETRIEVAL,
                    "similarity_threshold": Config.SIMILARITY_THRESHOLD,
                    "semantic_weight": Config.SEMANTIC_WEIGHT,
                    "keyword_weight": Config.KEYWORD_WEIGHT
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get retrieval stats: {str(e)}")
            return {"error": str(e)}
    
    def test_retrieval(self, test_query: str = "test") -> Dict[str, Any]:
        """Test the retrieval system with a simple query"""
        try:
            # Test all search types
            semantic_results = self.semantic_retrieve(test_query, 3)
            keyword_results = self.keyword_retrieve(test_query, 3)
            hybrid_results = self.hybrid_retrieve(test_query, 3)
            
            return {
                "test_query": test_query,
                "semantic_results_count": len(semantic_results),
                "keyword_results_count": len(keyword_results),
                "hybrid_results_count": len(hybrid_results),
                "total_documents": self.vector_store.collection.count() if self.vector_store.collection else 0,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Retrieval test failed: {str(e)}")
            return {
                "test_query": test_query,
                "status": "failed",
                "error": str(e)
            }