"""One SHAP worker per process, bounded waiting, no premature slot release on timeout."""

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import BoundedSemaphore, Event

from eris_ml.api_b2.schemas import ExplainResponse
from eris_ml.api_b2.service import ServiceError


class ExplanationWorker:
    def __init__(self, queue_size: int, timeout: float):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eris-shap")
        self._slots = BoundedSemaphore(1 + queue_size)
        self.timeout = timeout

    def _release(self, future: Future[ExplainResponse]) -> None:
        self._slots.release()

    async def run(self, call: Callable[[], ExplainResponse]) -> ExplainResponse:
        if not self._slots.acquire(blocking=False):
            raise ServiceError("EXPLANATION_BUSY", 429)
        abandoned = Event()

        def execute() -> ExplainResponse:
            if abandoned.is_set():
                raise ServiceError("EXPLANATION_TIMEOUT", 504)
            return call()

        try:
            future = self._executor.submit(execute)
        except RuntimeError:
            self._slots.release()
            raise ServiceError("MODEL_NOT_READY", 503) from None
        future.add_done_callback(self._release)
        wrapped = asyncio.wrap_future(future)
        # Consume late exceptions after the HTTP caller has timed out/disconnected.
        wrapped.add_done_callback(lambda f: None if f.cancelled() else f.exception())
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), self.timeout)
        except TimeoutError:
            # Keep even abandoned queued slots until dequeued: cancelling Futures here
            # would retain payloads in the executor queue while admitting unlimited retries.
            abandoned.set()
            raise ServiceError("EXPLANATION_TIMEOUT", 504) from None
        except asyncio.CancelledError:
            abandoned.set()
            raise

    def close(self) -> None:
        # No request body is persisted. Running work cannot safely be killed by a thread timeout.
        self._executor.shutdown(wait=True, cancel_futures=True)
