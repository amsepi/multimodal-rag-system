import os
from typing import Dict, Any

class Settings:
    """Configuration settings for the RAG system"""
    
    # PDF Processing Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))  # Reduced from 512 for better precision
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # Increased overlap
    MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))  # Minimum chunk length
    
    # Retrieval Settings
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "12"))  # Increased from 5
    IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", "8"))
    USE_HYBRID_SEARCH = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    
    # LLM Settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # Lower for more consistent answers
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))
    CONTEXT_MAX_TOKENS = int(os.getenv("CONTEXT_MAX_TOKENS", "6000"))
    
    # Embedding Settings
    TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
    
    # Scoring and Ranking
    METADATA_BOOST_WEIGHT = float(os.getenv("METADATA_BOOST_WEIGHT", "0.3"))
    CONTENT_OVERLAP_WEIGHT = float(os.getenv("CONTENT_OVERLAP_WEIGHT", "0.2"))
    LENGTH_SCORE_WEIGHT = float(os.getenv("LENGTH_SCORE_WEIGHT", "0.1"))
    VECTOR_SCORE_WEIGHT = float(os.getenv("VECTOR_SCORE_WEIGHT", "0.7"))
    
    # Quality Thresholds
    MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.5"))
    MIN_ANSWER_LENGTH = int(os.getenv("MIN_ANSWER_LENGTH", "100"))
    
    # Prompt Strategies
    DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "comprehensive")
    FALLBACK_STRATEGIES = ["comprehensive", "detailed", "step_by_step", "academic"]
    
    @classmethod
    def get_chunking_config(cls) -> Dict[str, Any]:
        """Get chunking configuration"""
        return {
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "min_chunk_length": cls.MIN_CHUNK_LENGTH
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """Get retrieval configuration"""
        return {
            "default_top_k": cls.DEFAULT_TOP_K,
            "image_top_k": cls.IMAGE_TOP_K,
            "use_hybrid_search": cls.USE_HYBRID_SEARCH,
            "enable_query_expansion": cls.ENABLE_QUERY_EXPANSION,
            "min_score_threshold": cls.MIN_SCORE_THRESHOLD
        }
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "model": cls.DEFAULT_MODEL,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS,
            "context_max_tokens": cls.CONTEXT_MAX_TOKENS,
            "default_strategy": cls.DEFAULT_STRATEGY,
            "fallback_strategies": cls.FALLBACK_STRATEGIES
        }

# Global settings instance
settings = Settings() 