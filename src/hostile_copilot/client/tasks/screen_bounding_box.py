import asyncio
import cv2
import logging
import numpy as np
import time
import pyautogui
import sys
from PIL.Image import Image

from hostile_copilot.config import OmegaConfig

from .base import Task
from .types import Coordinate

logger = logging.getLogger(__name__)

class GetScreenBoundingBoxTask(Task):
    def __init__(self, config: OmegaConfig):
        super().__init__(config)

        self._start: Coordinate | None = None
        self._end: Coordinate | None = None
        self._is_cropping: bool = False
        self._screenshot: Image | None = None
        self._cursor: Coordinate | None = None
        self._dirty: bool = False
        self._done: bool = False
        
        self._last_update: float = 0.0
        self._update_interval: float = 0.05
        self._alpha_mask: float = 0.40
        self._final_alpha_mask: float = 0.60
        self._final_delay: int = 1500
        # Overlay tint color (BGR). Slight red hint to differentiate.
        self._tint_color = (0, 0, 80)
        
    @property
    def start(self) -> Coordinate | None:
        return self._start
    
    @property
    def end(self) -> Coordinate | None:
        return self._end
    
    @property
    def bounding_box(self) -> tuple[Coordinate, Coordinate] | None:
        if self._start is None or self._end is None:
            return None
        return self._start, self._end

    async def run(self):
        await asyncio.to_thread(self.calibration_routine)

    def calibration_routine(self):
        try:
            print("Calibrating screen...")
            self._start = None
            self._end = None
            self._done = False
            self._dirty = True

            screenshot = pyautogui.screenshot()

            # Convert to CV2
            screenshot = self._convert_to_cv2(screenshot)
            self._screenshot = screenshot
            
            # Create window
            cv2.namedWindow("Screenshot", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Screenshot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Screenshot", screenshot)

            # Try to ensure window is on top / foreground
            try:
                # OpenCV 4.5+: set topmost if available
                cv2.setWindowProperty("Screenshot", cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            # Process initial events so window exists
            cv2.waitKey(1)
            # Windows-specific foreground enforcement
            self._bring_window_to_front("Screenshot")

            # Set mouse callback
            cv2.setMouseCallback("Screenshot", self.mouse_callback)

            # UI loop: repaint at fixed rate when dirty, and process events
            while not self._done:
                now = time.perf_counter()
                if self._dirty and (now - self._last_update) >= self._update_interval:
                    self.paint()
                    self._last_update = now
                    self._dirty = False

                # Pump events and allow ESC to cancel
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self._done = True

            # Final paint if selection completed
            if self._start is not None and self._end is not None:
                self.paint(final=True)
                cv2.waitKey(self._final_delay)
            cv2.destroyAllWindows()

            logger.debug(f"Calibrated screen: {self._start} to {self._end}")
        except Exception as e:
            logger.exception(f"Error calibrating screen: {e}")
        finally:
            # Reset state
            self._screenshot = None
            self._is_cropping = False
            self._cursor = None
            self._dirty = False
            self._done = False
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._start is None:
                self._start = Coordinate(x, y)
                self._is_cropping = True
                self._cursor = Coordinate(x, y)
                self._dirty = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._is_cropping:
                self._cursor = Coordinate(x, y)
                self._dirty = True
  
        elif event == cv2.EVENT_LBUTTONUP:
            self._is_cropping = False
            self._end = Coordinate(x, y)
            self._cursor = Coordinate(x, y)

            # Ensure coordinates are ordered correctly
            self._start, self._end = self._normalize_coords(self._start, self._end)

            # Request a final repaint and mark done
            self._dirty = True
            self._done = True

    def paint(self, final=False):
        screenshot = self._screenshot.copy()

        # Determine current rectangle
        rect_start = None
        rect_end = None
        if self._start is not None and self._end is not None:
            rect_start, rect_end = self._start, self._end
        elif self._is_cropping and self._start is not None and self._cursor is not None:
            rect_start, rect_end = self._start, self._cursor

        if rect_start is not None and rect_end is not None:
            # Normalize coordinates
            norm_start, norm_end = self._normalize_coords(rect_start, rect_end)
            x1, y1 = norm_start.x, norm_start.y
            x2, y2 = norm_end.x, norm_end.y

            # Create an overlay that's black everywhere, but with the region of interest
            # (ROI) copied from the original
            overlay = screenshot.copy()
            overlay[:] = self._tint_color
            overlay[y1:y2, x1:x2] = screenshot[y1:y2, x1:x2]

            # Blend: outside ROI becomes darker (15%), ROI stays original
            alpha_mask = self._alpha_mask if not final else self._final_alpha_mask
            cv2.addWeighted(overlay, alpha_mask, screenshot, 1 - alpha_mask, 0, screenshot)

            # Draw rectangle outline
            cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            # No ROI yet: dim the entire screen with a slight red tint to differentiate from the raw screenshot
            alpha_mask = self._alpha_mask if not final else self._final_alpha_mask
            overlay = np.full_like(screenshot, self._tint_color)
            cv2.addWeighted(overlay, alpha_mask, screenshot, 1 - alpha_mask, 0, screenshot)

        cv2.imshow("Screenshot", screenshot)
        
    def _normalize_coords(self, a: Coordinate, b: Coordinate) -> tuple[Coordinate, Coordinate]:
        """Return a tuple of Coordinates ordered as top-left (min) and bottom-right (max)."""
        x1, y1 = min(a.x, b.x), min(a.y, b.y)
        x2, y2 = max(a.x, b.x), max(a.y, b.y)
        return Coordinate(x1, y1), Coordinate(x2, y2)

    def _bring_window_to_front(self, window_name: str) -> None:
        """Best-effort: bring an OpenCV window to the foreground on Windows.

        - Sets the window as topmost briefly, then restores non-topmost to avoid lingering.
        - Uses Win32 APIs via ctypes. Does nothing on non-Windows platforms.
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)

            FindWindowW = user32.FindWindowW
            FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            FindWindowW.restype = wintypes.HWND

            SetForegroundWindow = user32.SetForegroundWindow
            SetForegroundWindow.argtypes = [wintypes.HWND]
            SetForegroundWindow.restype = wintypes.BOOL

            ShowWindow = user32.ShowWindow
            ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
            ShowWindow.restype = wintypes.BOOL

            SetWindowPos = user32.SetWindowPos
            SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
            SetWindowPos.restype = wintypes.BOOL

            HWND_TOPMOST = ctypes.c_void_p(-1)
            HWND_NOTOPMOST = ctypes.c_void_p(-2)
            SW_SHOW = 5
            SWP_NOSIZE = 0x0001
            SWP_NOMOVE = 0x0002
            SWP_SHOWWINDOW = 0x0040

            hwnd = FindWindowW(None, window_name)
            if hwnd:
                # Make visible and topmost briefly
                ShowWindow(hwnd, SW_SHOW)
                SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
                SetForegroundWindow(hwnd)
                # Optionally remove topmost flag to avoid sticking above everything
                SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        except Exception:
            # Best-effort only; ignore failures
            pass

    def _convert_to_cv2(self, image: Image) -> np.ndarray:
        image_data = np.array(image)

        if image_data.ndim == 3 and image_data.shape[2] == 4:
            cv_img = cv2.cvtColor(image_data, cv2.COLOR_RGBA2BGR)
        elif image_data.ndim == 3 and image_data.shape[2] == 3:
            cv_img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        else:  # grayscale or other mode
            temp_image = image.convert("RGB")
            image_data = np.array(temp_image)
            cv_img = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
        
        return cv_img