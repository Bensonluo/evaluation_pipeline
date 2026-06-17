"""
Chatbot API Client for RAG Evaluation Pipeline (black-box, HTTP only).

The RAG chatbot is treated as a black box: the only access path is the public
HTTP API at `POST /api/v1/chat`. We never touch Qdrant, the DB, or any internal
service. This keeps the evaluation honest — it measures what real users see.

Contract implemented here (verified against app/api/v1/chat.py):
- Request:  {"message": str, "session_id": int (>0), "user_id"?: int, "max_tokens"?: int}
- Response: {"content": str, "session_id": int, "intent": str,
             "sources": [str]|null, "metadata": {...}|null,
             "dialogue_state": {"phase","pending_slots","filled_slots"}|null}

Key behaviours captured from the black box:
- Auth is anonymous-friendly: omit the Authorization header entirely. The server
  falls back to user_id=0. Do NOT send a bad token (that 401s), and do NOT call
  POST /sessions (it 500s for anonymous clients). An arbitrary positive integer
  session_id works without pre-creation.
- session_id doubles as the conversation thread (in-memory MemorySaver keyed on
  thread_id). Reuse one session_id for multi-turn; use distinct ids per
  independent single-turn sample to avoid memory bleed.
- Global rate limit is ~10 req/min/IP. We throttle client-side by default to
  stay safely under it; raise `rate_limit_rpm` only when the server cap is
  raised.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Default chat path under the API v1 prefix.
CHAT_PATH = "/api/v1/chat"
HEALTH_PATH = "/health"


@dataclass
class GenerateResponse:
    """Response from chatbot generation API (new black-box contract).

    Mirrors the fields of `ChatResponse` in app/api/v1/chat.py so callers can
    evaluate on the real surface: content text, detected intent, retrieved
    sources, slot-filling state, and dialogue phase.
    """

    response: str  # alias of `content`; kept for backward-compat with callers
    latency_ms: int
    status: str = "success"
    error: Optional[str] = None
    intent: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    filled_slots: Dict[str, Any] = field(default_factory=dict)
    pending_slots: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    dialogue_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any], latency_ms: int) -> "GenerateResponse":
        """Build a GenerateResponse from the raw API JSON body."""
        metadata = data.get("metadata") or {}
        dialogue_state = data.get("dialogue_state") or {}
        # Sources may be absent (non-RAG intents); normalize to list.
        sources = data.get("sources") or []
        return cls(
            response=data.get("content", ""),
            latency_ms=latency_ms,
            status="success",
            intent=data.get("intent"),
            sources=list(sources),
            filled_slots=metadata.get("filled_slots") or {},
            pending_slots=metadata.get("pending_slots") or [],
            confidence=metadata.get("confidence"),
            dialogue_state=dialogue_state,
            metadata=metadata,
        )

    @classmethod
    def error_response(cls, error: str) -> "GenerateResponse":
        return cls(response="", latency_ms=0, status="error", error=error)


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
        successful_latencies = [
            r.latency_ms for r in self.responses if r.status == "success"
        ]
        return sum(successful_latencies) / len(successful_latencies)


class ChatbotClient:
    """Async black-box client for the RAG Chatbot HTTP API.

    Features:
    - Anonymous (no auth header) access; works with the server's demo mode.
    - Per-conversation session_id management for multi-turn evaluation.
    - Client-side rate limiting to stay under the server's ~10 req/min/IP cap.
    - Batch generation with concurrency control and retry/backoff.
    """

    DEFAULT_TIMEOUT = 60.0
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    # Stay safely under the server's in-memory 10 req/min/IP rate limit.
    DEFAULT_RATE_LIMIT_RPM = 9

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        api_key: Optional[str] = None,
        rate_limit_rpm: int = DEFAULT_RATE_LIMIT_RPM,
        chat_path: str = CHAT_PATH,
    ):
        """Initialize chatbot client.

        Args:
            base_url: Base URL of the chatbot API (e.g. "http://localhost:8000").
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts per request.
            api_key: Optional Bearer token. Omit for anonymous (demo) access.
            rate_limit_rpm: Max requests per minute the client will issue.
                Defaults to 9 to stay under the server's 10/min/IP limit.
            chat_path: Path to the chat endpoint (default /api/v1/chat).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.chat_path = chat_path
        self._client: Optional[httpx.AsyncClient] = None
        # Global token-bucket-style limiter shared across all requests on this
        # client so batch jobs don't blow past the per-IP cap.
        min_interval = 60.0 / max(rate_limit_rpm, 1)
        self._min_interval = min_interval
        self._limiter = asyncio.Lock()
        self._last_request_at = 0.0
        # Monotonic session id allocator for multi-turn conversations.
        self._session_counter = itertools.count(start=1)

    async def __aenter__(self) -> "ChatbotClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers=self._build_headers(),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers. No Authorization header unless a key is set."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def allocate_session_id(self) -> int:
        """Allocate a fresh, unique positive-integer session_id.

        Use one per conversation thread. The server does not validate these
        against its DB; it only uses them as the in-memory thread key.
        """
        return next(self._session_counter)

    async def _throttle(self) -> None:
        """Block until enough time has elapsed since the last request.

        Serializes on a lock so concurrent batch workers share one budget.
        """
        async with self._limiter:
            now = time.monotonic()
            wait = self._last_request_at + self._min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    async def generate(
        self,
        message: str,
        session_id: Optional[int] = None,
        user_id: Optional[int] = None,
        max_tokens: Optional[int] = None,
        *,
        query: Optional[str] = None,  # backward-compat alias for `message`
        context: Optional[Any] = None,  # ignored; kept for old callers
        conversation_history: Optional[Any] = None,  # ignored; kept for old callers
        **kwargs: Any,
    ) -> GenerateResponse:
        """Generate a response for a single message (black-box /api/v1/chat).

        Args:
            message: User message text (required). `query` is accepted as an
                alias for backward compatibility.
            session_id: Conversation thread id. If omitted, a fresh id is
                allocated so each call is an independent single-turn turn.
                Pass the same id across calls to keep a multi-turn thread.
            user_id: Optional user id; defaults to 0 server-side when omitted.
            max_tokens: Optional max response tokens.
            context, conversation_history: Ignored. Accepted only so legacy
                callers don't break. The black box derives context from its
                own retrieval pipeline.
            **kwargs: Extra fields merged into the request body.

        Returns:
            GenerateResponse with content, intent, sources, and slot state.
        """
        text = message if message is not None else query
        if not text:
            return GenerateResponse.error_response("message is required")

        # Fresh thread per call by default → no cross-sample memory bleed.
        sid = session_id if session_id is not None else self.allocate_session_id()

        payload: Dict[str, Any] = {
            "message": text,
            "session_id": sid,
        }
        if user_id is not None:
            payload["user_id"] = user_id
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(kwargs)

        client = self._get_or_create_client()
        last_error: Optional[str] = None

        for attempt in range(self.max_retries):
            await self._throttle()
            start_time = time.time()
            try:
                response = await client.post(self.chat_path, json=payload)
                latency_ms = int((time.time() - start_time) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    return GenerateResponse.from_api(data, latency_ms)

                # 429 means we still exceeded the server cap despite throttling.
                # Back off harder before retrying.
                if response.status_code == 429:
                    last_error = f"HTTP 429: rate limited ({response.text[:200]})"
                else:
                    last_error = (
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
            except httpx.TimeoutError as e:
                last_error = f"Timeout error: {e}"
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
            except Exception as e:  # noqa: BLE001 — surface as error response
                last_error = f"Unexpected error: {e}"

            if attempt < self.max_retries - 1:
                # Extra backoff on rate limiting.
                multiplier = 4.0 if "429" in (last_error or "") else 1.0
                delay = self.DEFAULT_RETRY_DELAY * (2**attempt) * multiplier
                logger.warning(
                    "Retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    delay,
                    last_error,
                )
                await asyncio.sleep(delay)

        return GenerateResponse.error_response(last_error or "Unknown error")

    async def batch_generate(
        self,
        messages: List[str],
        session_ids: Optional[List[int]] = None,
        max_concurrency: int = 3,
        **kwargs: Any,
    ) -> BatchGenerateResponse:
        """Generate responses for multiple messages with concurrency control.

        Concurrency is intentionally modest because the server limits ~10
        req/min/IP; the shared throttle enforces that regardless of the
        semaphore size.

        Args:
            messages: List of user messages.
            session_ids: Optional per-message session ids. If omitted, each
                message gets a fresh independent thread.
            max_concurrency: Max concurrent in-flight requests.
            **kwargs: Extra args forwarded to generate() (user_id, max_tokens).

        Returns:
            BatchGenerateResponse with all generated responses.
        """
        start_time = time.time()

        if session_ids is None:
            session_ids = [self.allocate_session_id() for _ in messages]
        elif len(session_ids) != len(messages):
            raise ValueError("Length of session_ids must match length of messages")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _one(idx: int) -> GenerateResponse:
            async with semaphore:
                return await self.generate(
                    message=messages[idx],
                    session_id=session_ids[idx],
                    **kwargs,
                )

        responses = await asyncio.gather(
            *[_one(i) for i in range(len(messages))],
            return_exceptions=True,
        )

        processed: List[GenerateResponse] = []
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error("Request %d failed: %s", i, resp)
                processed.append(GenerateResponse.error_response(str(resp)))
            else:
                processed.append(resp)

        total_latency_ms = int((time.time() - start_time) * 1000)
        success_count = sum(1 for r in processed if r.status == "success")
        failure_count = len(processed) - success_count

        return BatchGenerateResponse(
            responses=processed,
            total_latency_ms=total_latency_ms,
            success_count=success_count,
            failure_count=failure_count,
        )

    async def converse(
        self,
        turns: List[str],
        session_id: Optional[int] = None,
        **kwargs: Any,
    ) -> List[GenerateResponse]:
        """Run a multi-turn conversation reusing one session_id (thread).

        Use this for slot-filling / intent-switching evaluation: turns share
        the server's in-memory conversation state.

        Args:
            turns: Ordered list of user messages in the conversation.
            session_id: Thread id. Fresh id allocated if omitted.
            **kwargs: Extra args forwarded to generate() (user_id, max_tokens).

        Returns:
            One GenerateResponse per turn, in order.
        """
        sid = session_id if session_id is not None else self.allocate_session_id()
        results: List[GenerateResponse] = []
        for turn in turns:
            results.append(await self.generate(message=turn, session_id=sid, **kwargs))
        return results

    async def health_check(self) -> bool:
        """Check if the chatbot API is healthy (GET /health)."""
        client = self._get_or_create_client()
        try:
            response = await client.get(HEALTH_PATH)
            return response.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _get_or_create_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=self._build_headers(),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Utility function for quick usage
async def generate_response(
    base_url: str,
    message: str,
    session_id: Optional[int] = None,
    api_key: Optional[str] = None,
    *,
    query: Optional[str] = None,
) -> GenerateResponse:
    """Quick utility to generate a single response.

    Example:
        >>> resp = await generate_response(
        ...     "http://localhost:8000", "如何申请退款?"
        ... )
        >>> print(resp.response, resp.intent, resp.pending_slots)
    """
    text = message if message is not None else query
    async with ChatbotClient(base_url, api_key=api_key) as client:
        return await client.generate(message=text, session_id=session_id)
