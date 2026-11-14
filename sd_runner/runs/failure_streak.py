"""Consecutive generation failures, for pausing a queue that only fails.

A generation runs on a pool worker, and its failure is logged there and goes no
further: the run still counts as finished, and the next queued or staged
request starts. With the backend down, every request fails within seconds and
the whole queue is consumed that way. ``RunController._post_run`` reads this
count and pauses the queue once it reaches ``PAUSE_AFTER_FAILURES``.

Process-wide rather than per run or generator, because what fails is usually
the backend, which outlives both. Any success resets it, so does resuming the
queue.
"""

import threading
from typing import Optional

#: Consecutive failures at which the queue is paused.
PAUSE_AFTER_FAILURES = 3


class FailureStreak:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._last_error: Optional[str] = None

    def record_failure(self, error: BaseException) -> None:
        with self._lock:
            self._count += 1
            self._last_error = str(error)

    def record_success(self) -> None:
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._last_error = None

    def snapshot(self) -> tuple[int, Optional[str]]:
        """The count and the last error, read together."""
        with self._lock:
            return self._count, self._last_error


failure_streak = FailureStreak()
