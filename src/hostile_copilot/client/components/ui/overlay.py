from __future__ import annotations

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent

from .components.base import Drawable

class Overlay(QWidget):
    """Transparent overlay that renders any list of Drawables."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        
        self._drawables: list[Drawable] = []
        self.showFullScreen()

    def set_drawables(self, drawables: list[Drawable]) -> None:
        """Replace all drawables and repaint."""
        self._drawables = drawables
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            for drawable in self._drawables:
                try:
                    painter.setRenderHint(QPainter.Antialiasing, False)
                    drawable.draw(painter)
                except Exception as e:
                    print(f"Error drawing drawable: {e}")
        finally:
            painter.end()