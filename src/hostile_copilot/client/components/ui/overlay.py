from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPaintEvent, QScreen, QTransform
from PySide6.QtWidgets import QWidget

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

    def showOnScreen(self, screen: QScreen) -> None:
        # Ensure native window exists
        self.setAttribute(Qt.WA_NativeWindow)

        win = self.windowHandle()
        if win is not None:
            win.setScreen(screen)

        # Move/size widget to that screen
        geo = screen.geometry()
        self.setGeometry(geo)

        # NOW show fullscreen
        self.showFullScreen()


    def clear_drawables(self) -> None:
        """Clear all drawables and repaint."""
        self._drawables.clear()
        self.update()

    def add_drawables(self, drawables: list[Drawable]) -> None:
        """Add a list of drawables to the overlay."""
        self._drawables.extend(drawables)
        self.update()

    def set_drawables(self, drawables: list[Drawable]) -> None:
        """Replace all drawables and repaint."""
        self._drawables = drawables
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        dpr = self.screen().devicePixelRatio()

        # Make drawables use REAL physical screen coords
        transform = QTransform().scale(1.0 / dpr, 1.0 / dpr)
        painter.setWorldTransform(transform)
        painter.setRenderHint(QPainter.Antialiasing, True)

        try:
            for drawable in self._drawables:
                try:
                    drawable.draw(painter)
                except Exception as e:
                    print(f"Error drawing drawable: {e}")
        finally:
            painter.end()