from __future__ import annotations
from abc import ABC, abstractmethod
from PySide6.QtGui import QPainter

class Drawable(ABC):
    """
    Any entity that can be drawn on the screen overlay.
    """
    @abstractmethod
    def draw(self, painter: QPainter) -> None:
        raise NotImplementedError

