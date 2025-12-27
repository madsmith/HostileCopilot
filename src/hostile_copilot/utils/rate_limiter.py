from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    def __init__(self, interval: float, count: int = 1):
        self._interval = interval
        self._count = count
        self._history = deque(maxlen=count)

    def check(self) -> bool:
        current_time = time.time()

        # Expire old checks
        while self._history and self._history[0] < current_time - self._interval:
            self._history.popleft()

        if len(self._history) < self._count:
            self._history.append(current_time)
            return True

        return False

class RateLimiters:
    def __init__(self, default_interval: float):
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._default_interval = default_interval

    def configure(self, key: str, interval: float, count: int = 1):
        self._rate_limiters[key] = RateLimiter(interval, count)

    def get(self, key: str) -> RateLimiter:
        if key not in self._rate_limiters:
            self._rate_limiters[key] = RateLimiter(self._default_interval)
        return self._rate_limiters[key]

