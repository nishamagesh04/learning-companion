from typing import Optional, Literal
from pydantic import BaseModel, Field

class PDFUploadRequest(BaseModel):
    """Data payload associated with uploading a PDF resource."""
    title: str = Field(..., description="Title of the learning resource document")
    module_id: Optional[int] = Field(None, description="Associated module ID")
    programme_name: Optional[str] = Field(None, description="Course or Programme name")
    week: Optional[str] = Field(None, description="Week number/label")
    section: Optional[str] = Field(None, description="Section heading/name")
    topic: Optional[str] = Field(None, description="Topic covered")
    
    # Optional Pipeline overrides
    chunk_strategy: Optional[Literal["semantic", "fixed"]] = Field("semantic", description="Chunking strategy")
    max_chunk_size: Optional[int] = Field(800, description="Max chunk size in chars/tokens")
    chunk_overlap: Optional[int] = Field(150, description="Chunk overlap in chars/tokens")
    min_chunk_size: Optional[int] = Field(100, description="Min chunk size threshold")

class PDFReprocessRequest(BaseModel):
    """Payload to trigger reprocessing of an existing PDF resource."""
    resource_id: int = Field(..., description="Resource ID to reprocess")
    chunk_strategy: Optional[Literal["semantic", "fixed"]] = Field(None, description="New chunking strategy if overriding")
    max_chunk_size: Optional[int] = Field(None, description="New max chunk size if overriding")
    chunk_overlap: Optional[int] = Field(None, description="New chunk overlap if overriding")
    generate_embeddings: bool = Field(True, description="Whether to regenerate vector embeddings")
