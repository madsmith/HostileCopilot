from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPen

from .box import Box

@dataclass
class LabeledBox(Box):
    label: str = ""
    font_name: str = "Arial"
    font_size: int = 10
    font_color: QColor | tuple[int, int, int, int] | None = None
    font_opacity: float | None = None
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.font_color, tuple):
            assert len(self.font_color) == 4
            self.font_color = QColor(*self.font_color)
        elif self.font_color is None:
            # Set font_color to black if luminosity of self.color is too high
            luminosity = self.color.redF() * 0.2126 + self.color.greenF() * 0.7152 + self.color.blueF() * 0.0722
            if luminosity > 0.8:
                self.font_color = QColor(0, 0, 0, 255)
            else:
                self.font_color = QColor(255, 255, 255, 255)

        opacity = self.font_opacity or self.opacity
        self.font_color.setAlphaF(opacity)
    
    def draw(self, painter: QPainter) -> None:
        super().draw(painter)
        if not self.label:
            return
        
        font = QFont(self.font_name, self.font_size)
        painter.setFont(font)

        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(self.label)

        padding = (6, 1)
        bound_width = text_rect.width() + padding[0] * 2 + self.thickness // 2
        bound_height = text_rect.height() + padding[1] * 2

        # Background for label
        origin = (
            self.x1 - self.thickness // 2,
            self.y1 - self.thickness // 2,
        )

        painter.fillRect(
            origin[0],
            origin[1] - bound_height,
            bound_width,
            bound_height,
            self.color,
        )
        
        # Label text
        painter.setPen(self.font_color)
        painter.drawText(
            origin[0] + padding[0],
            origin[1] - padding[1] - 2, # Random padding that I can't explain
            self.label,
        )