from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import QWidget


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    color: QColor | None = None


class OverlayDetections(QWidget):
    """Full-screen transparent overlay for drawing detections.

    The window:
    - is borderless and always on top
    - has a transparent background
    - is click-through so it won't interfere with the game window
    """

    def __init__(self) -> None:
        super().__init__()

        # Frameless, always on top, no taskbar icon
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
        )

        # Transparent background and mouse click-through
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self._detections: List[Detection] = []

        self.showFullScreen()

    def update_detections(self, detections: List[Detection], margin: int = 0) -> None:
        """Replace current detections and trigger a repaint.

        Calling this clears the overlay logically (no accumulation).
        """
        self._detections = detections

        # Apply margin to all detections
        self._detections = [
            Detection(
                x1=d.x1 - margin,
                y1=d.y1 - margin,
                x2=d.x2 + margin,
                y2=d.y2 + margin,
                label=d.label,
                color=d.color,
            )
            for d in detections
        ]
        for det in self._detections:
            if det.color is None:
                print("!!!! No color for detection:", det)
                det.color = QColor(0, 200, 0, 255)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)

            pen = QPen(QColor(0, 255, 0, 255), 2)
            painter.setPen(pen)
            font = QFont("Arial", 10)
            painter.setFont(font)

            for det in self._detections:
                if det.color is None:
                    print("!!No color for detection:", det)
                    color = QColor(0, 200, 0, 255)
                else:
                    color = det.color
                painter.setPen(color)

                try:
                    x1 = int(det.x1)
                    y1 = int(det.y1)
                    x2 = int(det.x2)
                    y2 = int(det.y2)
                except Exception:
                    print("Failed to process detection:", det)
                    import traceback
                    traceback.print_exc()
                    continue

                # Skip obviously invalid or degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                rect = QRect(x1, y1, x2 - x1, y2 - y1)
                if rect.isNull() or rect.isEmpty():
                    continue

                painter.drawRect(rect)

                if det.label:
                    text_width = max(40, len(det.label) * 7)
                    text_height = 16
                    painter.fillRect(
                        x1,
                        y1 - text_height,
                        text_width,
                        text_height,
                        color,
                    )
                    # Set text to black
                    painter.setPen(QColor(0, 0, 0, 255))
                    painter.drawText(x1 + 3, y1 - 4, det.label)

        finally:
            painter.end()
