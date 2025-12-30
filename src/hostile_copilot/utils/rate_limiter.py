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

class NoopRateLimiter(RateLimiter):
    def __init__(self):
        super().__init__(0, 0)
    
    def check(self) -> bool:
        return True

class RateLimiters:
    def __init__(self, default_interval: float = 1.0):
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._default_interval = default_interval

    @classmethod
    def get_instance(cls) -> RateLimiters:
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    @classmethod
    def configure_limiter(cls, key: str, interval: float, count: int = 1):
        cls.get_instance().configure(key, interval, count)

    @classmethod
    def get_limiter(cls, key: str) -> RateLimiter:
        return cls.get_instance().get(key)

    def configure(self, key: str, interval: float, count: int = 1):
        if not interval:
            self._rate_limiters[key] = NoopRateLimiter()
        else:
            self._rate_limiters[key] = RateLimiter(interval, count)

    def get(self, key: str) -> RateLimiter:
        if key not in self._rate_limiters:
            self._rate_limiters[key] = RateLimiter(self._default_interval)
        return self._rate_limiters[key]

