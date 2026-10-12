"""Typed boundary for the threadpoolctl context manager used by V3 tuning."""

from contextlib import AbstractContextManager
from types import TracebackType

class threadpool_limits(AbstractContextManager["threadpool_limits"]):
    def __init__(self, limits: int) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
