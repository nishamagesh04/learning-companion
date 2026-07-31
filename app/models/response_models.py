from typing import Optional, List
import datetime
from pydantic import BaseModel, Field

class ExtractedPage(BaseModel):
    """Extracted text per page."""
    page_number: int
    text: str

class ExtractedContentPreview(BaseModel):
    """Preview response of raw vs cleaned extracted PDF text."""
    resource_id: int
    file_name: str
    extraction_method: str
    total_pages: int
    raw_text: str
    cleaned_text: str

class ChunkPreview(BaseModel):
    """Preview item of a single processed chunk."""
    chunk_index: int
    chunk_text: str
    token_count: int
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    has_embedding: bool = False

class ProcessingLogItem(BaseModel):
    """Individual log entry for pipeline auditing."""
    processing_stage: str
    status: str
    message: str
    created_at: datetime.datetime

class IngestionStatusResponse(BaseModel):
    """Overall status response for a PDF resource ingestion or reprocessing."""
    resource_id: int
    title: str
    file_name: str
    processing_status: str  # PENDING, EXTRACTING, CLEANING, CHUNKING, EMBEDDING, COMPLETED, FAILED
    processing_error: Optional[str] = None
    chunk_count: int = 0
    uploaded_at: datetime.datetime
    processed_at: Optional[datetime.datetime] = None
    logs: List[ProcessingLogItem] = []
