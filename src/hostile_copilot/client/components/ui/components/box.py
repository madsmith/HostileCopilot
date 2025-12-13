from PySide6.QtGui import QColor, QPen
from PySide6.QtCore import QRect
from PySide6.QtGui import QPainter

from .base import Drawable

@dataclass
class Box(Drawable):
    x1: int
    y1: int
    x2: int
    y2: int
    color: QColor | tuple[int, int, int, int] = (0, 255, 0, 255)
    thickness: int = 2
    
    def __post_init__(self):
        if isinstance(self.color, tuple):
            assert len(self.color) == 4
            self.color = QColor(*self.color)
    
    def draw(self, painter: QPainter) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            return

        painter.setPen(QPen(self.color, self.thickness))
        painter.drawRect(QRect(self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1))
