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
        self.font_color = self._validate_color(self.font_color, QColor(255, 255, 255, 255))

        # Apply opacity if provided
        if self.font_opacity is not None:
            self.font_color.setAlphaF(self.font_opacity)

    def _validate_color(self, color: QColor | tuple[int, int, int] | tuple[int, int, int, int] | None, default_color: QColor | None = None) -> QColor:
        if color is None:
            return default_color
        
        if isinstance(color, QColor):
            return color
        elif isinstance(color, tuple):
            if len(color) == 3:
                return QColor(*color, 255)
            elif len(color) == 4:
                return QColor(*color)
            else:
                raise ValueError(f"Invalid color tuple: {color}")
