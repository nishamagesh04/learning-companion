import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from app.models.database_models import LearningResource, ResourceContent, ProcessingLog
from app.models.request_models import PDFUploadRequest
from app.core.exceptions import DatabaseStorageError
from app.core.logging import get_logger

logger = get_logger("resource_repository")

class ResourceRepository:
    """Handles database persistence operations for LearningResource and ProcessingLog."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def create_resource(self, file_name: str, file_path: str, req: PDFUploadRequest) -> LearningResource:
        """Creates an initial LearningResource record with PENDING status."""
        try:
            resource = LearningResource(
                title=req.title,
                module_id=req.module_id,
                resource_type="PDF",
                file_name=file_name,
                file_path=file_path,
                programme_name=req.programme_name,
                week=req.week,
                section=req.section,
                topic=req.topic,
                processing_status="PENDING",
                version=1,
                is_active=True,
                uploaded_at=datetime.datetime.utcnow()
            )
            self.db.add(resource)
            self.db.commit()
            self.db.refresh(resource)
            logger.info(f"Created LearningResource record ID={resource.id} for file '{file_name}'.")
            return resource
        except Exception as e:
            self.db.rollback()
            raise DatabaseStorageError(f"Failed to create resource database record: {str(e)}") from e

    def update_processing_status(
        self,
        resource_id: int,
        status: str,
        error_message: Optional[str] = None
    ) -> LearningResource:
        """Updates status, error message, and processed_at timestamp."""
        try:
            resource = self.db.query(LearningResource).filter(LearningResource.id == resource_id).first()
            if resource:
                resource.processing_status = status
                if error_message is not None:
                    resource.processing_error = error_message
                if status == "COMPLETED":
                    resource.processed_at = datetime.datetime.utcnow()
                self.db.commit()
                self.db.refresh(resource)
            return resource
        except Exception as e:
            self.db.rollback()
            raise DatabaseStorageError(f"Failed to update status for resource ID={resource_id}: {str(e)}") from e

    def save_resource_content(
        self,
        resource_id: int,
        raw_text: str,
        cleaned_text: str,
        extraction_method: str = "pypdf"
    ) -> ResourceContent:
        """Saves raw and cleaned extracted text for a learning resource."""
        try:
            content = self.db.query(ResourceContent).filter(ResourceContent.resource_id == resource_id).first()
            if not content:
                content = ResourceContent(
                    resource_id=resource_id,
                    raw_text=raw_text,
                    cleaned_text=cleaned_text,
                    extraction_method=extraction_method,
                    created_at=datetime.datetime.utcnow()
                )
                self.db.add(content)
            else:
                content.raw_text = raw_text
                content.cleaned_text = cleaned_text
                content.extraction_method = extraction_method
            
            self.db.commit()
            self.db.refresh(content)
            return content
        except Exception as e:
            self.db.rollback()
            raise DatabaseStorageError(f"Failed to save extracted content for resource ID={resource_id}: {str(e)}") from e

    def add_processing_log(
        self,
        resource_id: int,
        stage: str,
        status: str,
        message: str
    ) -> ProcessingLog:
        """Records an audit log entry for processing stage progress/error."""
        try:
            log_entry = ProcessingLog(
                resource_id=resource_id,
                processing_stage=stage,
                status=status,
                message=message,
                created_at=datetime.datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            self.db.refresh(log_entry)
            return log_entry
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to write processing log entry: {str(e)}")

    def increment_resource_version(self, resource_id: int) -> LearningResource:
        """Increments version counter when reprocessing a resource."""
        resource = self.db.query(LearningResource).filter(LearningResource.id == resource_id).first()
        if resource:
            resource.version = (resource.version or 1) + 1
            self.db.commit()
            self.db.refresh(resource)
        return resource

    def get_resource_by_id(self, resource_id: int) -> Optional[LearningResource]:
        """Fetches resource by ID."""
        return self.db.query(LearningResource).filter(LearningResource.id == resource_id).first()

    def get_resource_content_preview(self, resource_id: int) -> Optional[ResourceContent]:
        """Fetches resource content for preview."""
        return self.db.query(ResourceContent).filter(ResourceContent.resource_id == resource_id).first()
