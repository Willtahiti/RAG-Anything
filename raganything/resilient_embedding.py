"""
Embedding (and LLM/vision) calls that survive server disconnections.

Indexing thousands of pages against a local Ollama/vLLM server means the
server will, at some point, restart, run out of memory, or be unreachable for
a while. ``make_resilient_embedding`` wraps any embedding function so that:

* transient failures (connection refused/reset, timeouts, 5xx, 429) are
  retried with exponential backoff and jitter, and when the server stays down
  the call *waits for it to come back* (optionally polling a health check) for
  up to ``max_outage_s`` instead of failing the whole document;
* permanent failures (4xx, bad arguments) are raised at once;
* every response is validated (one vector per text, expected dimension, finite
  values, no all-zero vectors) so a truncated or corrupt answer is never
  indexed. The vectors are returned as given, never truncated or rounded.

The same wrapper is available for LLM/vision callables via
``make_resilient_llm`` (no response validation, longer timeout).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence

import numpy as np

from raganything.resilience import _DEFAULT_RETRYABLE

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_RETRYABLE_NAME_PARTS = (
    "connect",
    "timeout",
    "disconnect",
    "unavailable",
    "overloaded",
    "ratelimit",
    "serverdisconnected",
    "remoteprotocol",
    "readerror",
    "incompleteread",
)


class EmbeddingUnavailableError(RuntimeError):
    """The embedding server stayed unreachable longer than ``max_outage_s``."""


class EmbeddingValidationError(ValueError):
    """The server answered, but the vectors are unusable."""


def _status_code(exc: BaseException) -> Optional[int]:
    for source in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status"):
            value = getattr(source, attr, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return None


def is_transient_error(exc: BaseException) -> bool:
    """Whether retrying (after a wait) can reasonably succeed."""
    if isinstance(exc, (EmbeddingValidationError,)):
        return True  # truncated answers from an overloaded server
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, _DEFAULT_RETRYABLE) or isinstance(exc, asyncio.TimeoutError):
        return True
    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    if isinstance(exc, (ValueError, TypeError, KeyError, AttributeError)):
        return False
    name = type(exc).__name__.lower()
    return any(part in name for part in _RETRYABLE_NAME_PARTS)


def validate_embeddings(
    result: Any, n_texts: int, expected_dim: Optional[int]
) -> np.ndarray:
    """Return ``result`` as a float32 matrix or raise EmbeddingValidationError."""
    try:
        array = np.asarray(result, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise EmbeddingValidationError(f"embeddings are not numeric: {exc}") from exc
    if array.ndim != 2:
        raise EmbeddingValidationError(
            f"expected a 2-D result, got shape {array.shape}"
        )
    if array.shape[0] != n_texts:
        raise EmbeddingValidationError(
            f"got {array.shape[0]} vectors for {n_texts} texts"
        )
    if expected_dim is not None and array.shape[1] != expected_dim:
        raise EmbeddingValidationError(
            f"vector dimension {array.shape[1]} != expected {expected_dim}"
        )
    if not np.isfinite(array).all():
        raise EmbeddingValidationError("embeddings contain NaN or infinite values")
    if n_texts and (np.abs(array).sum(axis=1) == 0).any():
        raise EmbeddingValidationError("embeddings contain an all-zero vector")
    return array


class ResilientCaller:
    """Retry/wait wrapper around an async callable."""

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        validate: Optional[Callable[[Any, tuple, dict], Any]] = None,
        request_timeout: Optional[float] = 120.0,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        max_outage_s: float = 3600.0,
        max_validation_retries: int = 3,
        health_check: Optional[Callable[[], Awaitable[bool]]] = None,
        health_interval: float = 15.0,
        max_concurrency: Optional[int] = None,
        on_outage: Optional[Callable[[BaseException], Any]] = None,
        on_recover: Optional[Callable[[float], Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        self._func = func
        self._validate = validate
        self.request_timeout = request_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_outage_s = max_outage_s
        self.max_validation_retries = max_validation_retries
        self.health_check = health_check
        self.health_interval = health_interval
        self.on_outage = on_outage
        self.on_recover = on_recover
        self.name = name or getattr(func, "__name__", "call")
        self._semaphore = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else None
        )
        self.stats: Dict[str, float] = {
            "calls": 0,
            "retries": 0,
            "outages": 0,
            "waited_s": 0.0,
            "validation_failures": 0,
        }

    def _delay(self, attempt: int) -> float:
        delay = min(
            self.base_delay * (2 ** min(max(attempt - 1, 0), 16)), self.max_delay
        )
        return delay * (1.0 + random.uniform(0, 0.25))

    async def _call_once(self, args: tuple, kwargs: dict) -> Any:
        coro = self._func(*args, **kwargs)
        if self.request_timeout:
            return await asyncio.wait_for(coro, self.request_timeout)
        return await coro

    async def _notify(self, callback: Optional[Callable], *args: Any) -> None:
        if callback is None:
            return
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # a UI callback must not break indexing
            logger.debug("resilience callback failed", exc_info=True)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._semaphore is None:
            return await self._run(args, kwargs)
        async with self._semaphore:
            return await self._run(args, kwargs)

    async def _run(self, args: tuple, kwargs: dict) -> Any:
        self.stats["calls"] += 1
        attempt = 0
        outage_started: Optional[float] = None
        validation_failures = 0
        while True:
            attempt += 1
            try:
                result = await self._call_once(args, kwargs)
                if self._validate is not None:
                    result = self._validate(result, args, kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not is_transient_error(exc):
                    raise
                if isinstance(exc, EmbeddingValidationError):
                    validation_failures += 1
                    self.stats["validation_failures"] += 1
                    if validation_failures > self.max_validation_retries:
                        raise
                else:
                    if outage_started is None:
                        outage_started = time.monotonic()
                        self.stats["outages"] += 1
                        logger.warning(
                            "%s: server unavailable (%s: %s); waiting for it to "
                            "come back (up to %.0fs)",
                            self.name,
                            type(exc).__name__,
                            exc,
                            self.max_outage_s,
                        )
                        await self._notify(self.on_outage, exc)
                    if time.monotonic() - outage_started > self.max_outage_s:
                        raise EmbeddingUnavailableError(
                            f"{self.name}: server unavailable for more than "
                            f"{self.max_outage_s:.0f}s (last error: "
                            f"{type(exc).__name__}: {exc}). Indexed data is kept; "
                            "restart the server and resume the ingestion."
                        ) from exc
                self.stats["retries"] += 1
                delay = self._delay(attempt)
                logger.warning(
                    "%s attempt %d failed (%s); retrying in %.1fs",
                    self.name,
                    attempt,
                    type(exc).__name__,
                    delay,
                )
                await asyncio.sleep(delay)
                if outage_started is not None:
                    await self._wait_until_healthy(outage_started)
                continue
            if outage_started is not None:
                waited = time.monotonic() - outage_started
                self.stats["waited_s"] += waited
                logger.info("%s: server is back after %.0fs", self.name, waited)
                await self._notify(self.on_recover, waited)
            return result

    async def _wait_until_healthy(self, outage_started: float) -> None:
        if self.health_check is None:
            return
        while time.monotonic() - outage_started <= self.max_outage_s:
            try:
                if await self.health_check():
                    return
            except Exception:
                pass
            await asyncio.sleep(self.health_interval)


def _as_function(caller: ResilientCaller, func: Callable) -> Callable:
    """Expose ``caller`` as a plain async function (what LightRAG expects)."""
    import functools

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        return await caller(*args, **kwargs)

    wrapper.stats = caller.stats  # type: ignore[attr-defined]
    wrapper.caller = caller  # type: ignore[attr-defined]
    return wrapper


def make_resilient_embedding(
    func: Callable[[Sequence[str]], Awaitable[Any]],
    *,
    expected_dim: Optional[int] = None,
    **options: Any,
) -> Callable[..., Awaitable[np.ndarray]]:
    """Wrap an async ``func(texts) -> vectors`` (see module docstring).

    ``options`` are the :class:`ResilientCaller` keyword arguments
    (``request_timeout``, ``max_outage_s``, ``health_check``, ...). The result
    is a plain async function with ``.stats`` and ``.caller`` attributes.
    """

    def validate(result: Any, args: tuple, kwargs: dict) -> np.ndarray:
        texts = args[0] if args else kwargs.get("texts", [])
        return validate_embeddings(result, len(texts), expected_dim)

    options.setdefault("name", getattr(func, "__name__", "embedding"))
    return _as_function(ResilientCaller(func, validate=validate, **options), func)


def make_resilient_llm(
    func: Callable[..., Awaitable[Any]], **options: Any
) -> Callable[..., Awaitable[Any]]:
    """Wrap an async LLM/vision function: same waiting, longer timeout."""
    options.setdefault("request_timeout", 900.0)
    options.setdefault("max_outage_s", 3600.0)
    options.setdefault("name", getattr(func, "__name__", "llm"))
    return _as_function(ResilientCaller(func, **options), func)
