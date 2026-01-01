from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtGui import QPainter, QFontMetrics, QFont

from .base import Drawable, TextComponent

Anchor = Literal["top_left", "top_center", "center"]


@dataclass
class TextBox(Drawable, TextComponent):
    x: int = 0
    y: int = 0
    text: str = ""
    anchor: Anchor | None = "top_left"

    def width(self) -> int:
        (width, _) = self._get_dimensions()
        return width
    
    def height(self) -> int:
        (_, height) = self._get_dimensions()
        return height

    def _get_dimensions(self) -> tuple[int, int]:
        font = QFont(self.font_name, self.font_size)
        
        metrics = QFontMetrics(font)
        lines = self.text.split("\n")
        if not lines:
            lines = [""]

        width = max(metrics.horizontalAdvance(line) for line in lines)
        height = (len(lines) - 1) * metrics.lineSpacing() + metrics.height()
        
        return (width, height)

    def draw(self, painter: QPainter) -> None:
        if not self.text:
            return

        # Ensure font settings applied
        font = QFont(self.font_name, self.font_size)
        painter.setFont(font)

        # Measure text with current painter font
        metrics = QFontMetrics(font)
        lines = self.text.split("\n")
        if not lines:
            lines = [""]
        text_width = max(metrics.horizontalAdvance(line) for line in lines)
        text_height = (len(lines) - 1) * metrics.lineSpacing() + metrics.height()

        # Default anchor is top-left
        draw_x = self.x
        draw_y = self.y

        if self.anchor == "center":
            draw_x = int(self.x - text_width / 2)
            top_y = self.y - (text_height / 2)
            draw_y = int(top_y + metrics.ascent())
        elif self.anchor == "top_center":
            draw_x = int(self.x - text_width / 2)
            draw_y = self.y + metrics.ascent()  # move to baseline from top
        else:  # "top_left" or None
            draw_y = self.y + metrics.ascent()  # move to baseline from top

        # Set pen color and draw
        painter.setPen(self.font_color)
        for i, line in enumerate(lines):
            painter.drawText(draw_x, draw_y + i * metrics.lineSpacing(), line)
