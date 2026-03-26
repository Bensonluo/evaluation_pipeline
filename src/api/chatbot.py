"""
Chatbot API Client for RAG Evaluation Pipeline.

Provides async client for:
- Generate responses
- Batch generation with concurrency control
- Latency tracking for performance monitoring
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GenerateResponse:
    """Response from chatbot generation API."""

    response: str
    latency_ms: int
    status: str = "success"
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchGenerateResponse:
    """Response from batch generation API."""

    responses: List[GenerateResponse]
    total_latency_ms: int
    success_count: int
    failure_count: int

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per successful request."""
        if self.success_count == 0:
            return 0.0
        successful_latencies = [r.latency_ms for r in self.responses if r.status == "success"]
        return sum(successful_latencies) / len(successful_latencies)


class ChatbotClient:
    """
    Async client for RAG Chatbot API.

    Features:
    - Async request handling with httpx
    - Automatic retry with exponential backoff
    - Latency tracking for performance monitoring
    - Batch generation with concurrency control
    """

    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_key: Optional[str] = None,
    ):
        """
        Initialize chatbot client.

        Args:
            base_url: Base URL of the chatbot API (e.g., "http://localhost:8000")
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key

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
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def generate(
        self,
        query: str,
        context: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> GenerateResponse:
        """
        Generate a response for a single query.

        Args:
            query: User query/question
            context: Optional list of context documents (for RAG)
            conversation_history: Optional conversation history
            **kwargs: Additional parameters to pass to the API

        Returns:
            GenerateResponse with the generated text and latency
        """
        client = self._get_or_create_client()

        payload = {
            "query": query,
            "context": context or [],
            "conversation_history": conversation_history or [],
            **kwargs,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await client.post("/api/chat", json=payload)
                latency_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    return GenerateResponse(
                        response=data.get("response", ""),
                        latency_ms=latency_ms,
                        status="success",
                        metadata=data.get("metadata", {}),
                    )
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

        return GenerateResponse(
            response="",
            latency_ms=0,
            status="error",
            error=last_error or "Unknown error",
        )

    async def batch_generate(
        self,
        queries: List[str],
        contexts: Optional[List[List[str]]] = None,
        max_concurrency: int = 10,
        **kwargs,
    ) -> BatchGenerateResponse:
        """
        Generate responses for multiple queries with concurrency control.

        Args:
            queries: List of user queries
            contexts: Optional list of context documents for each query
            max_concurrency: Maximum number of concurrent requests
            **kwargs: Additional parameters to pass to the API

        Returns:
            BatchGenerateResponse with all generated responses
        """
        start_time = time.time()

        # Prepare contexts (use empty list if not provided)
        if contexts is None:
            contexts = [[] for _ in queries]
        elif len(contexts) != len(queries):
            raise ValueError("Length of contexts must match length of queries")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def generate_with_semaphore(idx: int) -> GenerateResponse:
            """Generate response with semaphore control."""
            async with semaphore:
                return await self.generate(
                    query=queries[idx],
                    context=contexts[idx],
                    **kwargs,
                )

        # Execute all requests concurrently
        responses = await asyncio.gather(
            *[generate_with_semaphore(i) for i in range(len(queries))],
            return_exceptions=True,
        )

        # Handle any exceptions
        processed_responses = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Request {i} failed: {resp}")
                processed_responses.append(
                    GenerateResponse(
                        response="",
                        latency_ms=0,
                        status="error",
                        error=str(resp),
                    )
                )
            else:
                processed_responses.append(resp)

        total_latency_ms = int((time.time() - start_time) * 1000)
        success_count = sum(1 for r in processed_responses if r.status == "success")
        failure_count = len(processed_responses) - success_count

        return BatchGenerateResponse(
            responses=processed_responses,
            total_latency_ms=total_latency_ms,
            success_count=success_count,
            failure_count=failure_count,
        )

    async def health_check(self) -> bool:
        """
        Check if the chatbot API is healthy.

        Returns:
            True if API is accessible, False otherwise
        """
        client = self._get_or_create_client()
        try:
            response = await client.get("/health")
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
async def generate_response(
    base_url: str,
    query: str,
    context: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> GenerateResponse:
    """
    Quick utility function to generate a response.

    Example:
        >>> response = await generate_response(
        ...     "http://localhost:8000",
        ...     "How do I reset my password?"
        ... )
        >>> print(response.response)
    """
    async with ChatbotClient(base_url, api_key=api_key) as client:
        return await client.generate(query, context)
