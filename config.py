import os
from typing import List, Optional

class Config:
    """Configuration class for the RAG system"""
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Retrieval
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    KEYWORD_WEIGHT: float = 0.3
    SEMANTIC_WEIGHT: float = 0.7
    
    # Vector Database
    VECTOR_DB_PATH: str = "./data/vectordb"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Language Model
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-medium"
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    
    # OCR
    TESSERACT_CONFIG: str = "--oem 3 --psm 6"
    
    # File Upload
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    SUPPORTED_EXTENSIONS: List[str] = [".pdf", ".docx", ".pptx", ".txt", ".csv"]
    
    # API
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    
    # Logging
    LOG_DIR: str = "./logs"
    LOG_LEVEL: str = "INFO"
    
    # Confidence Scoring
    MIN_CONFIDENCE_THRESHOLD: float = 0.5
    LOW_CONFIDENCE_MESSAGE: str = "I'm not entirely confident in this answer. Please verify with the source documents."
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        os.makedirs(cls.VECTOR_DB_PATH, exist_ok=True)
        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs("./data", exist_ok=True)