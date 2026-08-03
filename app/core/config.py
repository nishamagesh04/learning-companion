from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal

class Settings:
    """Application and Ingestion Pipeline Configuration Settings."""
    
    # Environment & Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./learning_companion.db")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # PDF File Constraints (Scaled for 500+ Page Company Manuals)
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "250"))
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".md", ".csv"}
    
    # Chunking Configuration
    DEFAULT_CHUNK_STRATEGY: Literal["semantic", "fixed"] = os.getenv("DEFAULT_CHUNK_STRATEGY", "semantic")
    MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "800"))       # Max characters or tokens per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))         # Overlap between adjacent chunks
    MIN_CHUNK_SIZE: int = int(os.getenv("MIN_CHUNK_SIZE", "100"))       # Minimum chunk size threshold
    
    # Large Scale Batching & Rate-Limit Retries
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "30"))   # Mini-batch size for embedding API calls
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "5"))   # Max retries on rate limit (429)
    EMBEDDING_RETRY_BACKOFF: float = float(os.getenv("EMBEDDING_RETRY_BACKOFF", "2.0")) # Exponential backoff multiplier
    DB_BATCH_SIZE: int = int(os.getenv("DB_BATCH_SIZE", "100"))                # Bulk database insertion batch size
    PROGRESS_LOG_INTERVAL_PAGES: int = int(os.getenv("PROGRESS_LOG_INTERVAL_PAGES", "25")) # Progress log interval
    
    # Embedding Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # Log Level
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
