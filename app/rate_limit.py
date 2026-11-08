import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from app import config

WINDOW_SECONDS = 60


class SlidingWindowRateLimiter:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        window_seconds: int = WINDOW_SECONDS,
    ):
        self._clock = clock
        self._window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str, limit: int) -> bool:
        if limit == 0:
            return True
        now = self._clock()
        with self._lock:
            for key, timestamps in list(self._requests.items()):
                while timestamps and timestamps[0] <= now - self._window_seconds:
                    timestamps.popleft()
                if not timestamps:
                    del self._requests[key]
            requests = self._requests[identity]
            if len(requests) >= limit:
                return False
            requests.append(now)
            return True


message_rate_limiter = SlidingWindowRateLimiter()


def allow_message(identity: str) -> bool:
    return message_rate_limiter.allow(identity, config.RATE_LIMIT_PER_MINUTE)
