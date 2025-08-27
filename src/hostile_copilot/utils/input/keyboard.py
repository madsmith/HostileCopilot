import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Iterable

from .gremlin.keyboard import (
    Key,
    send_key_down,
    send_key_up,
    key_from_name,
    key_from_code,
)

logger = logging.getLogger(__name__)

# Public key-like type accepted by this module
KeyLikeT = Key | str | tuple[int, bool] | int


@dataclass
class _PressReq:
    key: Key
    interkey_delay: float  # time from previous press start to next press start
    press_duration: float  # time between key down and key up


class Keyboard:
    """
    Asynchronous keyboard that:
    - Preserves ordering of press starts via a FIFO queue.
    - Enforces inter-key delay between start times.
    - Allows overlapping presses by launching key-up as independent tasks.
    - Supports per-press randomized timing profiles.
    """

    def __init__(self):
        # Base timing (seconds)
        self._interkey_mean: float = 0.090
        self._interkey_std: float = 0.015
        self._press_mean: float = 0.080
        self._press_std: float = 0.015
        self._min_delay: float = 0.010

        # Worker state
        self._queue: asyncio.Queue[_PressReq] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._running: bool = False
        self._last_start: float = 0.0  # monotonic timestamp of last press start
        self._active: set[asyncio.Task] = set()
        # Track currently depressed keys; guarded by _pressed_lock
        self._pressed: set[Key] = set()
        self._pressed_lock = asyncio.Lock()
        self._pressed_cv = asyncio.Condition(self._pressed_lock)

    # ---------- Public API ----------
    async def start(self):
        if self._running:
            return
        self._running = True
        self._last_start = time.monotonic()
        self._worker_task = asyncio.create_task(self._worker(), name="Keyboard::worker")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        # Put a sentinel to unblock queue get
        await self._queue.put(_PressReq(key=Key("", 0, False, 0), interkey_delay=0, press_duration=0))
        # Cancel any in-flight presses and wait for them to finish
        if self._active:
            tasks = list(self._active)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._worker_task:
            try:
                await self._worker_task
            finally:
                self._worker_task = None

    def set_interkey_profile(self, mean: float, std: float, min_delay: float = 0.010):
        self._interkey_mean = max(min_delay, mean)
        self._interkey_std = max(0.0, std)
        self._min_delay = max(0.0, min_delay)

    def set_press_profile(self, mean: float, std: float, min_delay: float = 0.010):
        self._press_mean = max(min_delay, mean)
        self._press_std = max(0.0, std)

    async def press_key(
        self,
        key: KeyLikeT,
        interkey_delay: float | None = None,
        press_duration: float | None = None,
    ):
        """
        Schedule a key press. Start time ordering is preserved, but key releases may overlap
        with subsequent presses.
        """
        if not self._running:
            # Allow implicit start for convenience
            await self.start()

        key_obj = self._resolve_key(key)
        ik = interkey_delay if interkey_delay is not None else self._sample_delay(self._interkey_mean, self._interkey_std)
        pd = press_duration if press_duration is not None else self._sample_delay(self._press_mean, self._press_std)
        
        await self._queue.put(_PressReq(key=key_obj, interkey_delay=ik, press_duration=pd))

    async def type_sequence(self, keys: Iterable[KeyLikeT], interkey_delay: float | None = None, press_duration: float | None = None):
        for k in keys:
            await self.press_key(k, interkey_delay=interkey_delay, press_duration=press_duration)
    
    async def asyncSleep(self, seconds: float):
        await asyncio.sleep(seconds)
    
    # ---------- Internal ----------
    async def _worker(self):
        logger.debug("Keyboard worker started")
        while self._running:
            try:
                req = await self._queue.get()
                logger.debug(f"Dequeued: {req}")

                # Check for stop signal
                if not self._running:
                    break

                # Enforce start time based on last start + req.interkey_delay
                now = time.monotonic()
                planned_start = max(self._last_start + req.interkey_delay, now)
                sleep_s = planned_start - now
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                # Start press: wait until the key is actually pressed (key-down) before
                # allowing the next item to proceed. This preserves actual start ordering.
                started_evt = asyncio.Event()
                t = asyncio.create_task(
                    self._do_press(req, started_evt),
                    name=f"Keyboard::press::{getattr(req.key, 'name', 'key')}"
                )
                self._active.add(t)
                t.add_done_callback(self._active.discard)
                # Wait for the key to actually go down before moving on
                await started_evt.wait()
                self._last_start = time.monotonic()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Keyboard worker error: {e}")
                break

    async def _do_press(self, req: _PressReq, started_evt: asyncio.Event | None = None):
        sent_down = False
        try:
            # If key is already depressed, wait until it is released before pressing again
            async with self._pressed_cv:
                while req.key in self._pressed:
                    await self._pressed_cv.wait()
                logger.debug(f"Pressing key: {req.key}")
                send_key_down(req.key)
                self._pressed.add(req.key)
                sent_down = True
            if started_evt is not None:
                started_evt.set()
            await asyncio.sleep(max(self._min_delay, req.press_duration))
        except asyncio.CancelledError:
            # Propagate after ensuring key-up in finally
            if started_evt is not None and not started_evt.is_set():
                # Unblock worker if cancellation happens before keydown (best-effort)
                started_evt.set()
            raise
        except Exception:
            # Swallow unexpected exceptions; ensure key-up in finally
            if started_evt is not None and not started_evt.is_set():
                started_evt.set()
            pass
        finally:
            if sent_down:
                logger.debug(f"Releasing key: {req.key}")
                try:
                    send_key_up(req.key)
                except Exception:
                    pass
                finally:
                    async with self._pressed_cv:
                        self._pressed.discard(req.key)
                        self._pressed_cv.notify_all()

    def _sample_delay(self, mean: float, std: float) -> float:
        if std <= 0:
            return max(self._min_delay, mean)
        # Gaussian with clamp to minimum
        val = random.gauss(mean, std)
        return max(self._min_delay, val)

    # ---------- Key resolution ----------
    def _resolve_key(self, key_like: KeyLikeT) -> Key:
        if isinstance(key_like, Key):
            return key_like
        if isinstance(key_like, str):
            k = key_from_name(key_like)
            if k is None:
                raise ValueError(f"Unknown key name: {key_like}")
            return k
        if isinstance(key_like, tuple) and len(key_like) == 2 and isinstance(key_like[0], int) and isinstance(key_like[1], bool):
            return key_from_code(key_like[0], key_like[1])
        if isinstance(key_like, int):
            # Assume non-extended if only scan_code provided
            return key_from_code(key_like, False)
        raise TypeError(f"Unsupported key type: {type(key_like)}")