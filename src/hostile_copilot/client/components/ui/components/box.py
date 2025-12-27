from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtCore import QRect

from .base import Drawable

@dataclass
class Box(Drawable):
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    color: QColor | tuple[int, int, int, int] = (0, 255, 0, 255)
    opacity: float = 1.0
    thickness: int = 1
    
    def __post_init__(self):
        if isinstance(self.color, tuple):
            if len(self.color) == 3:
                self.color = QColor(*self.color, 255)
            elif len(self.color) == 4:
                self.color = QColor(*self.color)
            else:
                raise ValueError("Color tuple must have 3 or 4 elements")
        self.color.setAlphaF(self.opacity)
    
    def draw(self, painter: QPainter) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return

        painter.setPen(QPen(self.color, self.thickness))
        painter.drawRect(QRect(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1))
