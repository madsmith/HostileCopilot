from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF

from .base import Drawable

@dataclass
class OrientedBox(Drawable):
    center_x: float
    center_y: float
    width: float
    height: float
    angle_rad: float
    color: QColor | tuple[int, int, int, int] = (0, 255, 0, 255)
    opacity: float = 1.0
    thickness: int = 1

    def __post_init__(self):
        if isinstance(self.color, tuple):
            assert len(self.color) == 4
            self.color = QColor(*self.color)
        self.color.setAlphaF(self.opacity)

    def _corners(self) -> list[QPointF]:
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        cos_angle = cos(self.angle_rad)
        sin_angle = sin(self.angle_rad)
        local = [
            (-half_width, -half_height),
            ( half_width, -half_height),
            ( half_width,  half_height),
            (-half_width,  half_height),
        ]
        pts = []
        for x, y in local:
            rx = x * cos_angle - y * sin_angle + self.center_x
            ry = x * sin_angle + y * cos_angle + self.center_y
            pts.append(QPointF(rx, ry))
        return pts

    def draw(self, painter: QPainter) -> None:
        pts = self._corners()
        painter.setPen(QPen(self.color, self.thickness))
        painter.drawPolygon(QPolygonF(pts))
