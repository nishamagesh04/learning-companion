class IngestionPipelineError(Exception):
    """Base exception for all ingestion pipeline failures."""
    pass

class PDFValidationError(IngestionPipelineError):
    """Raised when PDF file validation fails (invalid format, corrupt header, exceeds size)."""
    pass

class ExtractionError(IngestionPipelineError):
    """Raised when PDF text extraction fails or returns unreadable empty text."""
    pass

class CleaningError(IngestionPipelineError):
    """Raised when text cleaning operations fail."""
    pass

class ChunkingError(IngestionPipelineError):
    """Raised when chunking strategy execution fails."""
    pass

class EmbeddingError(IngestionPipelineError):
    """Raised when vector embedding generation fails."""
    pass

class DatabaseStorageError(IngestionPipelineError):
    """Raised when storing resources, chunks, or logs into database fails."""
    pass
