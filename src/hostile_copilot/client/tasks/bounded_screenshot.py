import io
from pathlib import Path
from PIL.Image import Image
import pyautogui

from hostile_copilot.config import OmegaConfig

from .base import Task

class BoundedScreenshotTask(Task):
    def __init__(
        self,
        config: OmegaConfig,
        bounding_box: tuple[int, int, int, int]
    ):
        super().__init__(config)
        self._bounding_box: tuple[int, int, int, int] = bounding_box
        self._image: Image | None = None

    @property
    def image(self) -> Image | None:
        return self._image

    def binary_encoded(self, format: str = "PNG") -> bytes:
        buffer = io.BytesIO()
        format = format.upper()
        self._image.save(buffer, format=format)
        return buffer.getvalue()
    
    async def run(self):
        screenshot = pyautogui.screenshot()

        self._image = screenshot.crop(self._bounding_box)

        if self._config.get("app.screenshots.debug", False):
            save_path = Path(self._config.get("app.screenshots.path", "screenshots"))
            save_path.mkdir(parents=True, exist_ok=True)
            self._image.save(save_path / "screenshot.png")

        if self._config.get("app.screenshots.archive", False):
            unique_naame = Path(save_path / "archive" / "screenshot_" + time.strftime("%Y%m%d_%H%M%S") + ".png")
            unique_naame.parent.mkdir(parents=True, exist_ok=True)
            self._image.save(unique_naame)