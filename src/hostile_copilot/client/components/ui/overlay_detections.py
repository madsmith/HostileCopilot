from __future__ import annotations

from dataclasses import dataclass
from PySide6.QtGui import QColor

from .overlay import Overlay
from .components.labeled_box import LabeledBox

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    color: QColor | None = None

class DetectionBox(LabeledBox):
    def __init__(self, detection: Detection, **kwargs):
        super().__init__(
            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,
            label=detection.label,
            color=detection.color,
            **kwargs
        )
        self.detection = detection

class OverlayDetections(Overlay):
    """Full-screen transparent overlay for drawing detections.

    The window:
    - is borderless and always on top
    - has a transparent background
    - is click-through so it won't interfere with the game window
    """

    def __init__(self) -> None:
        super().__init__()

    def update_detections(self, detections: list[Detection], margin: int = 0) -> None:
        """Replace current detections and trigger a repaint.

        Calling this clears the overlay logically (no accumulation).
        """
        drawable_detections = [
            DetectionBox(
                Detection(
                    x1=d.x1 - margin,
                    y1=d.y1 - margin,
                    x2=d.x2 + margin,
                    y2=d.y2 + margin,
                    label=d.label,
                    color=d.color,
                ),
                font_opacity=0.5,
                opacity=0.2
            )
            for d in detections
        ]
        self.set_drawables(drawable_detections)