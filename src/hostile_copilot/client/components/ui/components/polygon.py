from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF

from .base import Drawable

@dataclass
class Polygon(Drawable):
    points: list[QPointF] | np.ndarray | list[tuple[float, float]]
    color: QColor | tuple[int, int, int, int] = (0, 255, 0, 255)
    opacity: float = 1.0
    thickness: int = 1
    
    def __post_init__(self):
        if isinstance(self.points, np.ndarray):
            self.points = [QPointF(x, y) for x, y in self.points]
        elif isinstance(self.points, list):
            self.points = [QPointF(x, y) for x, y in self.points]
            
        if isinstance(self.color, tuple):
            assert len(self.color) == 4
            self.color = QColor(*self.color)
        self.color.setAlphaF(self.opacity)
    
    def draw(self, painter: QPainter) -> None:
        painter.setPen(QPen(self.color, self.thickness))
        painter.drawPolygon(QPolygonF(self.points))