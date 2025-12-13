from PySide6.QtGui import QPainter, QColor, QFont

from .box import Box

@dataclass
class LabeledBox(Box):
    label: str = ""
    font_name: str = "Arial"
    font_size: int = 10
    font_color: QColor | tuple[int, int, int, int] = (0, 0, 0, 255)
    
    def __post_init__(self):
        if isinstance(self.font_color, tuple):
            assert len(self.font_color) == 4
            self.font_color = QColor(*self.font_color)
    
    def draw(self, painter: QPainter) -> None:
        super().draw(painter)
        if not self.label:
            return
        
        font = QFont(self.font_name, self.font_size)

        # Background for label
        text_width = max(40, len(self.label) * font.pointSize() / 2)
        text_height = font.pointSize() + 4
        painter.fillRect(self.x1, self.y1 - text_height, text_width, text_height, self.color)
        
        # Label text
        painter.setPen(self.font_color)
        painter.setFont(font)
        painter.drawText(self.x1 + 3, self.y1 - 4, self.label)