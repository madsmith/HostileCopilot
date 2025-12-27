from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, TextIO


@dataclass
class _SectionStats:
    count: int = 0
    total_sec: float = 0.0
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def add(self, duration: float) -> None:
        self.count += 1
        self.total_sec += duration
        if duration < self.min_sec:
            self.min_sec = duration
        if duration > self.max_sec:
            self.max_sec = duration

    def average(self) -> float:
        return self.total_sec / self.count if self.count else 0.0


class Profiler:
    """
    Simple, thread-safe profiler for timing code blocks.

    Usage:
        with Profiler("section_name"):
            some_block_of_code()

        Profiler.dump()
        print("section name took " + Profiler.get_average("section_name"))
    """

    _lock: threading.Lock = threading.Lock()
    _stats: Dict[str, _SectionStats] = {}
    _enabled: bool = False
    _track_nested: bool = False
    _local: threading.local = threading.local()

    def __init__(self, section: str):
        self.section = section
        self._start: Optional[float] = None
        self._effective_section: str = section

    def __enter__(self) -> Profiler:
        if Profiler._enabled:
            if Profiler._track_nested:
                stack = getattr(Profiler._local, "stack", None)
                if stack is None:
                    stack = []
                    Profiler._local.stack = stack
                if stack:
                    self._effective_section = ".".join([*stack, self.section])
                else:
                    self._effective_section = self.section
                stack.append(self.section)
            else:
                self._effective_section = self.section

            self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not Profiler._enabled:
            return
        if self._start is None:
            return
        end = time.perf_counter()
        duration = end - self._start
        with Profiler._lock:
            stats = Profiler._stats.get(self._effective_section)
            if stats is None:
                stats = _SectionStats()
                Profiler._stats[self._effective_section] = stats
            stats.add(duration)

        if Profiler._track_nested:
            stack = getattr(Profiler._local, "stack", None)
            if stack:
                stack.pop()

    # ===== Controls =====
    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled
    
    @classmethod
    def enable(cls, enabled: bool = True) -> None:
        cls._enabled = bool(enabled)

    @classmethod
    def track_nested(cls, enabled: bool = True) -> None:
        cls._track_nested = bool(enabled)

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._stats.clear()


    # ===== Queries =====
    @classmethod
    def get_average_seconds(cls, section_prefix: str, *, exact: bool = False) -> float:
        with cls._lock:
            items = cls._stats.items()
            if exact:
                stats = cls._stats.get(section_prefix)
                return stats.average() if stats else 0.0

            matched = [
                st
                for name, st in items
                if (
                    name == section_prefix
                    or name.startswith(section_prefix + ".")
                    or name.endswith("." + section_prefix)
                    or ("." + section_prefix + ".") in name
                )
            ]
            if not matched:
                return 0.0

            total_count = sum(st.count for st in matched)
            total_sec = sum(st.total_sec for st in matched)
            return (total_sec / total_count) if total_count else 0.0

    @classmethod
    def get_stats(cls, section_prefix: str, *, exact: bool = False) -> Dict[str, _SectionStats]:
        with cls._lock:
            if exact:
                stats = cls._stats.get(section_prefix)
                if stats is None:
                    return {}
                return {section_prefix: _SectionStats(stats.count, stats.total_sec, stats.min_sec, stats.max_sec)}

            results: Dict[str, _SectionStats] = {}
            for name, stats in cls._stats.items():
                if (
                    name == section_prefix
                    or name.startswith(section_prefix + ".")
                    or name.endswith("." + section_prefix)
                    or ("." + section_prefix + ".") in name
                ):
                    results[name] = _SectionStats(stats.count, stats.total_sec, stats.min_sec, stats.max_sec)
            return results

    @classmethod
    def dump(cls, file: Optional[TextIO] = None) -> None:
        """Print a stats table sorted by total time."""
        out = file or sys.stdout
        with cls._lock:
            items = sorted(cls._stats.items(), key=lambda kv: kv[1].total_sec, reverse=True)

        print("Profiler Stats:", file=out)
        if not items:
            print("  (no data)", file=out)
            return

        header = f"{'section':30}  {'count':>8}  {'total':>10}  {'avg':>10}  {'min':>10}  {'max':>10}"
        print(header, file=out)
        print("-" * len(header), file=out)
        for name, st in items:
            avg = st.average()
            print(
                f"{name:30}  {st.count:8d}  {st.total_sec:10.3f}s  {avg:10.3f}s  {st.min_sec:10.3f}s  {st.max_sec:10.3f}s",
                file=out,
            )

