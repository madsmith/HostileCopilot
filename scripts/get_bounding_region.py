import sys
import time
import json
import cv2
import numpy as np
import pyautogui


class Coordinate:
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)


class BoundingRegionSelector:
    def __init__(self):
        self._start: Coordinate | None = None
        self._end: Coordinate | None = None
        self._cursor: Coordinate | None = None
        self._is_dragging: bool = False
        self._done: bool = False
        self._dirty: bool = True
        self._update_interval: float = 0.05
        self._last_update: float = 0.0
        self._alpha_mask: float = 0.40
        self._final_alpha_mask: float = 0.60
        self._tint_color = (0, 0, 80)  # BGR slight red tint
        self._window_name = "Select Region"
        self._screenshot: np.ndarray | None = None

    def run(self) -> tuple[Coordinate, Coordinate] | None:
        shot = pyautogui.screenshot()
        self._screenshot = self._convert_to_cv2(shot)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self._window_name, self._screenshot)

        try:
            cv2.setWindowProperty(self._window_name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.waitKey(1)
        self._bring_window_to_front(self._window_name)

        cv2.setMouseCallback(self._window_name, self._mouse_callback)

        while not self._done:
            now = time.perf_counter()
            if self._dirty and (now - self._last_update) >= self._update_interval:
                self._paint()
                self._last_update = now
                self._dirty = False

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC cancels
                self._start = None
                self._end = None
                self._done = True

        if self._start is not None and self._end is not None:
            # Final paint
            self._paint(final=True)
            cv2.waitKey(500)

        cv2.destroyAllWindows()

        if self._start is None or self._end is None:
            return None
        return self._start, self._end

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._start = Coordinate(x, y)
            self._cursor = Coordinate(x, y)
            self._is_dragging = True
            self._dirty = True
        elif event == cv2.EVENT_MOUSEMOVE and self._is_dragging:
            self._cursor = Coordinate(x, y)
            self._dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            self._is_dragging = False
            self._end = Coordinate(x, y)
            self._cursor = Coordinate(x, y)
            if self._start is not None and self._end is not None:
                self._start, self._end = self._normalize_coords(self._start, self._end)
            self._dirty = True
            self._done = True

    def _paint(self, final: bool = False):
        img = self._screenshot.copy()

        rect_start = None
        rect_end = None
        if self._start is not None and self._end is not None:
            rect_start, rect_end = self._start, self._end
        elif self._is_dragging and self._start is not None and self._cursor is not None:
            rect_start, rect_end = self._start, self._cursor

        if rect_start is not None and rect_end is not None:
            a, b = self._normalize_coords(rect_start, rect_end)
            x1, y1, x2, y2 = a.x, a.y, b.x, b.y

            overlay = img.copy()
            overlay[:] = self._tint_color
            overlay[y1:y2, x1:x2] = img[y1:y2, x1:x2]

            alpha = self._final_alpha_mask if final else self._alpha_mask
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            alpha = self._final_alpha_mask if final else self._alpha_mask
            overlay = np.full_like(img, self._tint_color)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        cv2.imshow(self._window_name, img)

    @staticmethod
    def _normalize_coords(a: Coordinate, b: Coordinate) -> tuple[Coordinate, Coordinate]:
        x1, y1 = min(a.x, b.x), min(a.y, b.y)
        x2, y2 = max(a.x, b.x), max(a.y, b.y)
        return Coordinate(x1, y1), Coordinate(x2, y2)

    @staticmethod
    def _bring_window_to_front(window_name: str) -> None:
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
                ShowWindow(hwnd, SW_SHOW)
                SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
                SetForegroundWindow(hwnd)
                SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
        except Exception:
            pass

    @staticmethod
    def _convert_to_cv2(pil_image) -> np.ndarray:
        arr = np.array(pil_image)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # grayscale or other
        return cv2.cvtColor(pil_image.convert("RGB"), cv2.COLOR_RGB2BGR)


def main() -> int:
    selector = BoundingRegionSelector()
    result = selector.run()
    if result is None:
        print("Selection cancelled", file=sys.stderr)
        return 1

    (a, b) = result
    x1, y1 = a.x, a.y
    x2, y2 = b.x, b.y
    w, h = (x2 - x1), (y2 - y1)

    payload = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": w, "height": h}
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
