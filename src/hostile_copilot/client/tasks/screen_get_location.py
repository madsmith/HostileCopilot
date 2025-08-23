import asyncio
from pynput import mouse

from hostile_copilot.client.tasks.base import Task
from hostile_copilot.config import OmegaConfig

from .types import Coordinate

class GetScreenLocationTask(Task):
    def __init__(self, config: OmegaConfig):
        super().__init__(config)
        self._last_click: Coordinate | None = None

    @property
    def last_click(self) -> Coordinate | None:
        return self._last_click

    async def run(self):
        await asyncio.to_thread(self._locate_click)

    def _locate_click(self):
        self._last_click = None

        def on_click(x, y, button, pressed):
            if pressed:
                self._last_click = Coordinate(x, y)
                return False  # stop listener

        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
