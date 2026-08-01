from dotenv import load_dotenv

load_dotenv()
import hashlib
import time
from typing import List, Callable, Optional
from app.core.config import settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger("embedding_client")

class GeminiEmbeddingClient:
    """Wrapper for Google Gemini Embedding API supporting mini-batching & exponential backoff retries."""

    def __init__(self, api_key: str = settings.GEMINI_API_KEY, model_name: str = settings.EMBEDDING_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self.dimension = settings.EMBEDDING_DIMENSION
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Gemini client for embedding model '{self.model_name}'.")
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai client ({str(e)}). Falling back to mock embedding generator.")

    def generate_single_embedding_with_retry(self, text: str) -> List[float]:
        """Generates embedding for a single text chunk with exponential backoff retry logic."""
        if not text or not text.strip():
            return [0.0] * self.dimension

        if self.client:
            max_retries = settings.EMBEDDING_MAX_RETRIES
            backoff = settings.EMBEDDING_RETRY_BACKOFF
            
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=text
                    )
                    if response and hasattr(response, 'embedding') and hasattr(response.embedding, 'values'):
                        return list(response.embedding.values)
                    elif hasattr(response, 'embeddings') and len(response.embeddings) > 0:
                        return list(response.embeddings[0].values)
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str
                    if is_rate_limit and attempt < max_retries:
                        sleep_time = backoff ** attempt
                        logger.warning(f"Rate limit hit on attempt {attempt}/{max_retries}. Backing off for {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Gemini API embedding call failed on attempt {attempt}: {err_str}. Using fallback generator.")
                        break

        # Deterministic mock embedding fallback for local dev / testing / fallback
        return self._generate_mock_embedding(text)

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[float]]:
        """
        Generates embeddings for a large list of texts using mini-batching.
        Prevents memory spikes and API rate limit breaches on 500+ page documents (1,500+ chunks).
        """
        total = len(texts)
        embeddings: List[List[float]] = []

        logger.info(f"Processing {total} text chunks in mini-batches of {batch_size}...")

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            for text in batch_texts:
                emb = self.generate_single_embedding_with_retry(text)
                embeddings.append(emb)

            current_count = len(embeddings)
            if progress_callback:
                progress_callback(current_count, total)
            elif (i // batch_size) % 5 == 0 or current_count == total:
                pct = int((current_count / total) * 100)
                logger.info(f"Embedding Progress: {current_count}/{total} chunks embedded ({pct}%).")

        return embeddings

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generates a normalized 768-dimensional float vector deterministically from text SHA-256 hash."""
        seed_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        values: List[float] = []
        for i in range(self.dimension):
            char_hex = seed_hash[(i % len(seed_hash))]
            val = (int(char_hex, 16) - 7.5) / 7.5
            values.append(val)
        
        # Normalize vector to unit length
        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        return values
