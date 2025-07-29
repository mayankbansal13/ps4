import os
import logging
from typing import List, Dict, Any, Optional
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class LLMIntegration:
    """Language model integration with OpenAI and HuggingFace fallback"""
    
    def __init__(self):
        self.openai_available = False
        self.hf_model = None
        self.hf_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available language models"""
        # Try OpenAI first
        if Config.OPENAI_API_KEY:
            try:
                openai.api_key = Config.OPENAI_API_KEY
                # Test the connection with a simple request
                response = openai.ChatCompletion.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                self.openai_available = True
                logger.info("OpenAI API initialized successfully")
            except Exception as e:
                logger.warning(f"OpenAI API not available: {str(e)}")
                self.openai_available = False
        else:
            logger.info("No OpenAI API key provided, using HuggingFace fallback")
        
        # Initialize HuggingFace model as fallback
        if not self.openai_available:
            try:
                logger.info(f"Loading HuggingFace model: {Config.HUGGINGFACE_MODEL}")
                self.hf_tokenizer = AutoTokenizer.from_pretrained(Config.HUGGINGFACE_MODEL)
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    Config.HUGGINGFACE_MODEL,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                # Add padding token if not present
                if self.hf_tokenizer.pad_token is None:
                    self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
                
                logger.info(f"HuggingFace model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load HuggingFace model: {str(e)}")
                raise
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer based on the question and retrieved context"""
        
        # Prepare context from chunks
        context_text = self._prepare_context(context_chunks)
        
        # Create the prompt
        prompt = self._create_prompt(question, context_text, context_chunks)
        
        # Generate answer using available model
        if self.openai_available:
            answer_data = self._generate_with_openai(prompt, question, context_chunks)
        else:
            answer_data = self._generate_with_huggingface(prompt, question, context_chunks)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(question, context_chunks, answer_data.get("answer", ""))
        answer_data["confidence"] = confidence
        
        # Add low confidence warning if needed
        if confidence < Config.MIN_CONFIDENCE_THRESHOLD:
            answer_data["warning"] = Config.LOW_CONFIDENCE_MESSAGE
        
        return answer_data
    
    def _prepare_context(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved chunks"""
        if not context_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks[:Config.TOP_K_RETRIEVAL]):
            metadata = chunk["metadata"]
            content = chunk["content"]
            
            # Add source information
            source_info = f"Source: {metadata['source']}"
            if metadata.get('file_type') == 'pdf' and 'Page' in content:
                # Extract page info if available
                source_info += " (from PDF page information in content)"
            
            context_part = f"[Context {i+1}] {source_info}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, question: str, context: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Create a detailed prompt for answer generation"""
        
        sources_list = []
        for chunk in context_chunks[:Config.TOP_K_RETRIEVAL]:
            source = chunk["metadata"]["source"]
            if source not in sources_list:
                sources_list.append(source)
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context. Your task is to provide accurate, factual answers that are directly supported by the given context.

INSTRUCTIONS:
1. Answer the question using ONLY the information provided in the context below
2. If the context doesn't contain enough information to answer the question, clearly state this
3. Always cite which source(s) your answer comes from
4. Be precise and avoid making assumptions beyond what's stated in the context
5. If there are conflicting information in different sources, mention this
6. Keep your answer clear, concise, and well-structured

CONTEXT:
{context}

QUESTION: {question}

SOURCES AVAILABLE: {', '.join(sources_list)}

ANSWER:"""
        
        return prompt
    
    def _generate_with_openai(self, prompt: str, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate answers based on given context. Always cite your sources and be truthful about what information is available."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE,
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "answer": answer,
                "model_used": "openai",
                "model_name": Config.OPENAI_MODEL,
                "sources": self._extract_sources(context_chunks),
                "token_usage": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            # Fallback to HuggingFace
            self.openai_available = False
            return self._generate_with_huggingface(prompt, question, context_chunks)
    
    def _generate_with_huggingface(self, prompt: str, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate answer using HuggingFace model"""
        try:
            # Encode the prompt
            inputs = self.hf_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # Decode the response
            generated_text = self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer
            answer = self._clean_generated_answer(answer)
            
            return {
                "answer": answer,
                "model_used": "huggingface",
                "model_name": Config.HUGGINGFACE_MODEL,
                "sources": self._extract_sources(context_chunks),
                "token_usage": len(outputs[0])
            }
            
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {str(e)}")
            return {
                "answer": "I apologize, but I'm unable to generate an answer due to a technical issue. Please try again or contact support.",
                "model_used": "fallback",
                "model_name": "error_fallback",
                "sources": self._extract_sources(context_chunks),
                "error": str(e)
            }
    
    def _clean_generated_answer(self, answer: str) -> str:
        """Clean up generated answer from HuggingFace model"""
        # Remove common artifacts
        answer = answer.replace("<|endoftext|>", "")
        answer = answer.replace("<pad>", "")
        
        # Split by common stop sequences and take the first part
        stop_sequences = ["\n\nQUESTION:", "\n\nCONTEXT:", "\n\nSOURCES:", "Human:", "Assistant:"]
        for stop_seq in stop_sequences:
            if stop_seq in answer:
                answer = answer.split(stop_seq)[0]
        
        # Limit length and clean up
        sentences = answer.split('. ')
        if len(sentences) > 10:  # Limit to reasonable length
            answer = '. '.join(sentences[:10]) + '.'
        
        return answer.strip()
    
    def _extract_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context chunks"""
        sources = []
        for chunk in context_chunks:
            metadata = chunk["metadata"]
            source_info = {
                "filename": metadata["source"],
                "file_type": metadata["file_type"],
                "chunk_id": metadata["chunk_id"]
            }
            
            # Add additional metadata if available
            if metadata.get("ocr_processed"):
                source_info["note"] = "Processed with OCR"
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence(self, question: str, context_chunks: List[Dict[str, Any]], answer: str) -> float:
        """Calculate confidence score for the generated answer"""
        if not context_chunks or not answer:
            return 0.0
        
        confidence_factors = []
        
        # Factor 1: Number of relevant chunks
        num_chunks = len(context_chunks)
        chunk_factor = min(num_chunks / Config.TOP_K_RETRIEVAL, 1.0)
        confidence_factors.append(chunk_factor * 0.3)
        
        # Factor 2: Average similarity scores
        similarity_scores = []
        for chunk in context_chunks:
            if "similarity_score" in chunk:
                similarity_scores.append(chunk["similarity_score"])
            elif "hybrid_score" in chunk:
                similarity_scores.append(chunk["hybrid_score"])
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            confidence_factors.append(avg_similarity * 0.4)
        else:
            confidence_factors.append(0.2)  # Low confidence if no scores
        
        # Factor 3: Answer length and structure (basic heuristic)
        answer_length_factor = min(len(answer.split()) / 50, 1.0)  # Normalize around 50 words
        confidence_factors.append(answer_length_factor * 0.2)
        
        # Factor 4: Presence of source citations in answer
        citation_factor = 0.0
        source_names = [chunk["metadata"]["source"] for chunk in context_chunks]
        for source in source_names:
            if source.lower() in answer.lower():
                citation_factor = 1.0
                break
        confidence_factors.append(citation_factor * 0.1)
        
        # Calculate final confidence
        final_confidence = sum(confidence_factors)
        return min(max(final_confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded models"""
        return {
            "openai_available": self.openai_available,
            "openai_model": Config.OPENAI_MODEL if self.openai_available else None,
            "huggingface_model": Config.HUGGINGFACE_MODEL,
            "device": self.device,
            "models_loaded": {
                "openai": self.openai_available,
                "huggingface": self.hf_model is not None
            }
        }