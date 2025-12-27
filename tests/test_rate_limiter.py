import pytest

from hostile_copilot.utils.rate_limiter import RateLimiter, RateLimiters


class TestRateLimiter:
    def test_burst_count_3_allows_3_then_denies_until_interval(self, monkeypatch: pytest.MonkeyPatch):
        t = 1000.0

        def fake_time() -> float:
            return t

        monkeypatch.setattr("hostile_copilot.utils.rate_limiter.time.time", fake_time)

        limiter = RateLimiter(interval=10.0, count=3)

        assert limiter.check() is True
        assert limiter.check() is True
        assert limiter.check() is True

        assert limiter.check() is False
        assert limiter.check() is False

        t += 9.999
        assert limiter.check() is False

        t += 0.002
        assert limiter.check() is True
        assert limiter.check() is True
        assert limiter.check() is True
        assert limiter.check() is False


class TestRateLimiters:
    def test_unconfigured_get_creates_default_limiter(self, monkeypatch: pytest.MonkeyPatch):
        t = 2000.0

        def fake_time() -> float:
            return t

        monkeypatch.setattr("hostile_copilot.utils.rate_limiter.time.time", fake_time)

        rate_limiters = RateLimiters(default_interval=1.0)
        limiter_a1 = rate_limiters.get("a")
        limiter_a2 = rate_limiters.get("a")

        assert limiter_a1 is limiter_a2

        assert limiter_a1.check() is True
        assert limiter_a1.check() is False

        t += 1.001
        assert limiter_a1.check() is True

    def test_configured_limiter_burst_count_3(self, monkeypatch: pytest.MonkeyPatch):
        t = 3000.0

        def fake_time() -> float:
            return t

        monkeypatch.setattr("hostile_copilot.utils.rate_limiter.time.time", fake_time)

        rate_limiters = RateLimiters(default_interval=999.0)
        rate_limiters.configure("api", interval=10.0, count=3)
        limiter = rate_limiters.get("api")

        assert limiter.check() is True
        assert limiter.check() is True
        assert limiter.check() is True

        assert limiter.check() is False
        assert limiter.check() is False

        t += 10.001
        assert limiter.check() is True
