from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from PySide6.QtGui import QPainter, QColor

class Drawable(ABC):
    """
    Any entity that can be drawn on the screen overlay.
    """
    @abstractmethod
    def draw(self, painter: QPainter) -> None:
        raise NotImplementedError


@dataclass
class TextComponent:
    font_name: str = "Arial"
    font_size: int = 10
    font_color: QColor | tuple[int, int, int, int] | None = None
    font_opacity: float | None = None

    def __post_init__(self):
        if isinstance(self.font_color, tuple):
            assert len(self.font_color) == 4
            self.font_color = QColor(*self.font_color)
        elif self.font_color is None:
            # Default to opaque white; components may override based on background
            self.font_color = QColor(255, 255, 255, 255)
        # Apply opacity if provided
        if self.font_opacity is not None:
            self.font_color.setAlphaF(self.font_opacity)
