"""
Qdrant Vector Database Client for RAG Evaluation Pipeline.

Provides async client for:
- Vector search (similarity search)
- Batch search operations
- Latency tracking for performance monitoring
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import logging

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    """Response from Qdrant search API."""

    doc_id: str
    score: float
    payload: Dict[str, Any]
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "score": self.score,
            "payload": self.payload,
            "latency_ms": self.latency_ms,
        }


@dataclass
class BatchSearchResponse:
    """Response from batch search API."""

    results: List[List[SearchResponse]]  # One list per query
    total_latency_ms: int
    query_count: int


@dataclass
class QdrantConfig:
    """Qdrant client configuration."""

    host: str
    port: int
    collection_name: str
    api_key: Optional[str] = None
    https: bool = False
    timeout: float = 30.0

    @property
    def base_url(self) -> str:
        scheme = "https" if self.https else "http"
        return f"{scheme}://{self.host}:{self.port}"


class QdrantClient:
    """
    Async client for Qdrant vector database.

    Features:
    - Vector similarity search
    - Batch search with concurrency control
    - Latency tracking for performance monitoring
    - Automatic retry with exponential backoff
    """

    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 0.5
    DEFAULT_LIMIT = 5  # Top-K results

    def __init__(
        self,
        base_url: str,
        collection_name: str,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize Qdrant client.

        Args:
            base_url: Base URL of Qdrant instance (e.g., "http://localhost:6333")
            collection_name: Name of the collection to search
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure HTTP client
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers=self._build_headers(),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["api-key"] = self.api_key
        return headers

    async def search(
        self,
        query_vector: List[float],
        limit: int = DEFAULT_LIMIT,
        score_threshold: Optional[float] = None,
        with_payload: Union[bool, List[str]] = True,
        **kwargs,
    ) -> List[SearchResponse]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return (Top-K)
            score_threshold: Optional minimum score threshold
            with_payload: Whether to include payload in results
            **kwargs: Additional search parameters

        Returns:
            List of SearchResponse objects sorted by score (descending)
        """
        client = self._get_or_create_client()

        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": with_payload,
            **kwargs,
        }

        if score_threshold is not None:
            payload["score_threshold"] = score_threshold

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await client.post(
                    f"/collections/{self.collection_name}/points/search",
                    json=payload,
                )
                latency_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    results = data.get("result", [])

                    return [
                        SearchResponse(
                            doc_id=str(result.get("id", "")),
                            score=float(result.get("score", 0.0)),
                            payload=result.get("payload", {}),
                            latency_ms=latency_ms,
                        )
                        for result in results
                    ]
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"

            except httpx.TimeoutError as e:
                last_error = f"Timeout error: {e}"
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
            except Exception as e:
                last_error = f"Unexpected error: {e}"

            # Retry with exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.DEFAULT_RETRY_DELAY * (2**attempt)
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay}s: {last_error}")
                await asyncio.sleep(delay)

        logger.error(f"Search failed after {self.max_retries} attempts: {last_error}")
        return []

    async def batch_search(
        self,
        query_vectors: List[List[float]],
        limit: int = DEFAULT_LIMIT,
        score_threshold: Optional[float] = None,
        max_concurrency: int = 10,
        **kwargs,
    ) -> BatchSearchResponse:
        """
        Search for multiple query vectors with concurrency control.

        Args:
            query_vectors: List of query embedding vectors
            limit: Maximum number of results per query
            score_threshold: Optional minimum score threshold
            max_concurrency: Maximum number of concurrent requests
            **kwargs: Additional search parameters

        Returns:
            BatchSearchResponse with results for all queries
        """
        start_time = time.time()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def search_with_semaphore(vector: List[float]) -> List[SearchResponse]:
            """Search with semaphore control."""
            async with semaphore:
                return await self.search(
                    query_vector=vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    **kwargs,
                )

        # Execute all searches concurrently
        results = await asyncio.gather(
            *[search_with_semaphore(v) for v in query_vectors],
            return_exceptions=True,
        )

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search {i} failed: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)

        total_latency_ms = int((time.time() - start_time) * 1000)

        return BatchSearchResponse(
            results=processed_results,
            total_latency_ms=total_latency_ms,
            query_count=len(query_vectors),
        )

    async def search_by_id(
        self,
        point_id: str,
        limit: int = DEFAULT_LIMIT,
        **kwargs,
    ) -> List[SearchResponse]:
        """
        Search for similar vectors by point ID (recommend search).

        Args:
            point_id: ID of the reference point
            limit: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of SearchResponse objects
        """
        client = self._get_or_create_client()

        payload = {
            "limit": limit,
            "with_payload": True,
            **kwargs,
        }

        try:
            start_time = time.time()
            response = await client.post(
                f"/collections/{self.collection_name}/points/{point_id}/recommend",
                json=payload,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                results = data.get("result", [])

                return [
                    SearchResponse(
                        doc_id=str(result.get("id", "")),
                        score=float(result.get("score", 0.0)),
                        payload=result.get("payload", {}),
                        latency_ms=latency_ms,
                    )
                    for result in results
                ]
        except Exception as e:
            logger.error(f"Recommend search failed: {e}")

        return []

    async def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the collection.

        Returns:
            Collection info dict or None if failed
        """
        client = self._get_or_create_client()

        try:
            response = await client.get(f"/collections/{self.collection_name}")
            if response.status_code == 200:
                return response.json().get("result", {})
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")

        return None

    async def health_check(self) -> bool:
        """
        Check if Qdrant is accessible.

        Returns:
            True if Qdrant is accessible, False otherwise
        """
        client = self._get_or_create_client()
        try:
            response = await client.get("/")
            return response.status_code == 200
        except Exception:
            return False

    def _get_or_create_client(self) -> httpx.AsyncClient:
        """Get existing client or create a new one."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._build_headers(),
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Utility function for quick usage
async def search_vectors(
    base_url: str,
    collection_name: str,
    query_vector: List[float],
    limit: int = 5,
    api_key: Optional[str] = None,
) -> List[SearchResponse]:
    """
    Quick utility function to search vectors.

    Example:
        >>> results = await search_vectors(
        ...     "http://localhost:6333",
        ...     "rag_chatbot_kb",
        ...     query_embedding,
        ...     limit=5
        ... )
        >>> for result in results:
        ...     print(f"{result.doc_id}: {result.score}")
    """
    async with QdrantClient(base_url, collection_name, api_key=api_key) as client:
        return await client.search(query_vector, limit=limit)
