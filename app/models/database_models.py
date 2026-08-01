import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class LearningResource(Base):
    """Stores uploaded resource metadata (Section 9.3)."""
    __tablename__ = 'learning_resources'

    id = Column(Integer, primary_key=True, autoincrement=True)
    module_id = Column(Integer, nullable=True)
    title = Column(String(255), nullable=False)
    resource_type = Column(String(50), nullable=False, default="PDF")
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    source_url = Column(String(500), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    processing_status = Column(String(50), default="PENDING", nullable=False) # PENDING, EXTRACTING, CLEANING, CHUNKING, EMBEDDING, COMPLETED, FAILED
    processing_error = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)

    # Metadata fields (Section 8.2)
    programme_name = Column(String(255), nullable=True)
    week = Column(String(50), nullable=True)
    section = Column(String(255), nullable=True)
    topic = Column(String(255), nullable=True)

    # Relationships
    content = relationship("ResourceContent", back_populates="resource", cascade="all, delete-orphan", uselist=False)
    chunks = relationship("ContentChunk", back_populates="resource", cascade="all, delete-orphan")
    logs = relationship("ProcessingLog", back_populates="resource", cascade="all, delete-orphan")


class ResourceContent(Base):
    """Stores raw and cleaned extracted text for auditing (Section 9.4)."""
    __tablename__ = 'resource_content'

    id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(Integer, ForeignKey('learning_resources.id', ondelete='CASCADE'), nullable=False, unique=True)
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    extraction_method = Column(String(100), default="pypdf", nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    resource = relationship("LearningResource", back_populates="content")


class ContentChunk(Base):
    """Stores chunked content and generated vector embeddings (Section 9.5)."""
    __tablename__ = 'content_chunks'

    id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(Integer, ForeignKey('learning_resources.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False, default=0)
    page_number = Column(Integer, nullable=True)
    start_timestamp = Column(String(50), nullable=True)
    end_timestamp = Column(String(50), nullable=True)
    section_heading = Column(String(255), nullable=True)
    embedding = Column(JSON, nullable=True)  # Stores vector embedding list float[]
    embedding_model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    resource = relationship("LearningResource", back_populates="chunks")


class ProcessingLog(Base):
    """Stores detailed processing events and errors for operational audit (Section 9.11)."""
    __tablename__ = 'processing_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(Integer, ForeignKey('learning_resources.id', ondelete='CASCADE'), nullable=False)
    processing_stage = Column(String(50), nullable=False) # VALIDATION, EXTRACTION, CLEANING, CHUNKING, EMBEDDING, PERSISTENCE, REPROCESSING
    status = Column(String(20), nullable=False)            # INFO, SUCCESS, WARNING, ERROR
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    resource = relationship("LearningResource", back_populates="logs")

class ConversationSession(Base):
    __tablename__ = 'conversation_sessions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey('conversation_sessions.id', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False)
    message_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    session = relationship("ConversationSession", back_populates="messages")
    response = relationship("ChatResponse", back_populates="message", uselist=False, cascade="all, delete-orphan")


class ChatResponse(Base):
    __tablename__ = 'chat_responses'
    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=False, unique=True)
    answer_text = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    message = relationship("ChatMessage", back_populates="response")