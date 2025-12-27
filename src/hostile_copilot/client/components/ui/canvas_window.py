import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import (
    QImage,
    QPaintEvent,
    QPainter,
    QWheelEvent,
    QTransform,
    QScreen,
    QMouseEvent,
    QCursor,
)
from PySide6.QtCore import QRectF, QSize, Qt

from .components import Drawable


class CanvasWindow(QWidget):
    """Simple window that renders the current frame and draws Drawables over it."""

    def __init__(self, title="Canvas") -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._image: QImage | None = None
        self._drawables: list[Drawable] = []
        self._img_width: int = 0
        self._img_height: int = 0
        self._scale: float = 1.0
        self._fit_scale: float = 1.0
        self._zoom: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._hint_screen_width: int = 0
        self._hint_screen_height: int = 0
        # Drag state
        self._dragging: bool = False
        self._last_mouse_x: float = 0.0
        self._last_mouse_y: float = 0.0

    def showOnScreen(self, screen: QScreen) -> None:
        self._hint_screen_width = screen.size().width()
        self._hint_screen_height = screen.size().height()

        # Ensure the widget has a native window handle BEFORE first show
        # so we can bind it to the target screen reliably.
        self.setAttribute(Qt.WA_NativeWindow, True)
        # Force creation of a native window handle
        _ = self.winId()
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(screen)

        # Place the window within the target screen's available geometry
        geo = screen.availableGeometry()

        # Default window size to ~60% of screen dimensions
        width = max(1, int(geo.width() * 0.6))
        height = max(1, int(geo.height() * 0.6))

        x = geo.x() + max(0, (geo.width() - width) // 2)
        y = geo.y() + max(0, (geo.height() - height) // 2)
        self.setGeometry(x, y, width, height)

        # Now show on that screen
        self.show()
        self.raise_()
        self.activateWindow()

    def screen_width(self) -> int:
        screen = self.screen()
        if screen is None:
            dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
            return int(round(self.width() * dpr))
        dpr = screen.devicePixelRatio()
        return int(round(screen.geometry().width() * dpr))

    def screen_height(self) -> int:
        screen = self.screen()
        if screen is None:
            dpr = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
            return int(round(self.height() * dpr))
        dpr = screen.devicePixelRatio()
        return int(round(screen.geometry().height() * dpr))

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        # Convert BGR (cv2) to RGB for Qt
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888)
        # Ensure a deep copy because source buffer may be reused
        self._image = qimg.copy()
        # Track source image size
        self._img_width, self._img_height = width, height
        # If the window has no explicit size yet, start at image size
        if not self.isVisible() or (self.width() == 0 and self.height() == 0):
            self.resize(width, height)
        # Recompute layout for new image
        self._compute_layout()
        self._clamp_offset()
        self.update()

    def _clamp_offset(self) -> None:
        """Clamp offsets so panning stays within allowable bounds.
        - If image larger than window: prevent exposing empty space beyond edges.
        - If image smaller (letterboxed): allow movement within the letterbox range.
        """
        window_width = max(1, self.width())
        window_height = max(1, self.height())
        image_width = self._img_width * self._scale if self._img_width else 0
        image_height = self._img_height * self._scale if self._img_height else 0

        min_x = min(0.0, window_width - image_width)
        max_x = max(0.0, window_width - image_width)
        min_y = min(0.0, window_height - image_height)
        max_y = max(0.0, window_height - image_height)

        self._offset_x = max(min_x, min(max_x, self._offset_x))
        self._offset_y = max(min_y, min(max_y, self._offset_y))

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._image is None:
            return
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_mouse_x = event.position().x()
            self._last_mouse_y = event.position().y()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if not self._dragging or self._image is None:
            return
        mx = event.position().x()
        my = event.position().y()
        dx = mx - self._last_mouse_x
        dy = my - self._last_mouse_y
        self._last_mouse_x = mx
        self._last_mouse_y = my
        # Apply delta directly in window-space since we translate before scaling
        self._offset_x += dx
        self._offset_y += dy
        self._clamp_offset()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.unsetCursor()

    def clear_drawables(self) -> None:
        """Clear all drawables and repaint."""
        self._drawables.clear()
        self.update()

    def add_drawable(self, drawable: Drawable) -> None:
        """Add a single drawable to the overlay."""
        self._drawables.append(drawable)
        self.update()

    def add_drawables(self, drawables: list[Drawable]) -> None:
        """Add a list of drawables to the overlay."""
        self._drawables.extend(drawables)
        self.update()
        
    def set_drawables(self, drawables: list[Drawable]) -> None:
        self._drawables = drawables
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            if self._image is not None:
                # Draw scaled image with letterboxing to preserve aspect
                dest_rect = QRectF(self._offset_x, self._offset_y, self._img_width * self._scale, self._img_height * self._scale)
                painter.drawImage(dest_rect, self._image)

            if self._drawables and self._image is not None:
                # Scale and translate to match image scaling for drawables
                transform = QTransform()
                transform.translate(self._offset_x, self._offset_y)
                transform.scale(self._scale, self._scale)
                painter.setWorldTransform(transform, combine=False)
                for drawable in self._drawables:
                    try:
                        drawable.draw(painter)
                    except Exception as e:
                        print(f"Error drawing drawable: {e}")
        finally:
            painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image is None or self._img_width == 0 or self._img_height == 0:
            return
        # Zoom step per wheel notch (120 units per notch). Exponential for smoothness.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        zoom_step = 1.1
        factor = zoom_step ** (delta / 120.0)

        # Compute min/max zoom. Min is fit-to-window (1.0). Max is arbitrary large.
        min_zoom = 1.0
        max_zoom = 20.0

        old_zoom = self._zoom
        new_zoom = max(min_zoom, min(max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # Anchor zoom around mouse position so the point under cursor stays put.
        mx = event.position().x()
        my = event.position().y()

        # Total scale before change
        old_scale = self._fit_scale * old_zoom

        # Image coords under the cursor
        ix = (mx - self._offset_x) / max(1e-6, old_scale)
        iy = (my - self._offset_y) / max(1e-6, old_scale)

        # Apply zoom and recompute total scale
        self._zoom = new_zoom
        self._scale = self._fit_scale * self._zoom

        # Reposition so (ix, iy) maps back under the cursor
        self._offset_x = mx - ix * self._scale
        self._offset_y = my - iy * self._scale
        self._clamp_offset()
        self.update()

    def resizeEvent(self, event) -> None:
        self._compute_layout()
        return super().resizeEvent(event)

    def sizeHint(self) -> QSize:
        if self._img_width and self._img_height:
            return QSize(self._img_width, self._img_height)
        if self._hint_screen_width and self._hint_screen_height:
            return QSize(self._hint_screen_width, self._hint_screen_height)
        return QSize(1920, 1080)

    def _compute_layout(self) -> None:
        if self._image is None or self._img_width == 0 or self._img_height == 0:
            self._scale = 1.0
            self._offset_x = 0.0
            self._offset_y = 0.0
            return
            
        window_width = max(1, self.width())
        window_height = max(1, self.height())
        scale_width = window_width / self._img_width
        scale_height = window_height / self._img_height
        self._fit_scale = min(scale_width, scale_height)
        self._scale = self._fit_scale * self._zoom

        # Center only when not zoomed; if zoomed, keep current offsets
        if abs(self._zoom - 1.0) < 1e-6:
            draw_width = self._img_width * self._scale
            draw_height = self._img_height * self._scale
            self._offset_x = (window_width - draw_width) / 2.0
            self._offset_y = (window_height - draw_height) / 2.0
