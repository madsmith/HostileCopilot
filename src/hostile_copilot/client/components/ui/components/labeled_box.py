from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPen

from .box import Box
from .base import TextComponent

@dataclass
class LabeledBox(Box, TextComponent):
    label: str = ""
    label_color: QColor | None = None
    
    def __post_init__(self):
        # Initialize Box first (sets color/opacity)
        Box.__post_init__(self)

        # Initialize TextComponent to normalize font attributes
        TextComponent.__post_init__(self)

        self.label_color: QColor | None = self._validate_color(self.label_color)

        # If no explicit font_color was provided, pick readable color based on box color luminosity
        if self.label_color is None:
            luminosity = self.color.redF() * 0.2126 + self.color.greenF() * 0.7152 + self.color.blueF() * 0.0722
            self.label_color = QColor(0, 0, 0, 255) if luminosity > 0.8 else QColor(255, 255, 255, 255)
        
        # Apply opacity fallback to box opacity when font_opacity is not set
        opacity = self.font_opacity or self.opacity
        self.font_color.setAlphaF(opacity)
        self.label_color.setAlphaF(opacity)
    
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
        painter.setPen(self.label_color)
        painter.drawText(
            origin[0] + padding[0],
            origin[1] - padding[1] - 2, # Random padding that I can't explain
            self.label,
        )