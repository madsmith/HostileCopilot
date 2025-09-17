import asyncio
import win32api, win32con, win32gui

class Overlay:
    def __init__(self):
        # Maintain a list of rectangles to render
        self.rects = []

        hInstance = win32api.GetModuleHandle(None)
        className = "OverlayWindow"

        wndClass = win32gui.WNDCLASS()
        wndClass.lpfnWndProc = self.wndProc
        wndClass.hInstance = hInstance
        wndClass.lpszClassName = className
        win32gui.RegisterClass(wndClass)

        self.hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST | 
            win32con.WS_EX_TOOLWINDOW |
            win32con.WS_EX_NOACTIVATE,
            className,
            None,
            win32con.WS_POPUP,
            0, 0,
            win32api.GetSystemMetrics(0),
            win32api.GetSystemMetrics(1),
            0, 0, hInstance, None
        )

        # Make black transparent
        win32gui.SetLayeredWindowAttributes(self.hwnd, 0x000000, 255, win32con.LWA_COLORKEY)

    def show(self):
        """Show the overlay window."""
        win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
        win32gui.UpdateWindow(self.hwnd)

    def hide(self):
        """Hide the overlay window."""
        win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)

    def add_rect(self, x: int, y: int, w: int, h: int):
        """Add a rectangle to be drawn and trigger a repaint."""
        self.rects.append((x, y, w, h))
        # Invalidate the entire window to force a repaint
        win32gui.InvalidateRect(self.hwnd, None, True)
        win32gui.UpdateWindow(self.hwnd)

    async def run(self):
        """Run the overlay message loop asynchronously until the window is closed.

        This method pumps the Win32 message queue without blocking the asyncio loop.
        Cancel the task or close the window to exit.
        """
        PM_REMOVE = 0x0001
        self.show()
        try:
            while True:
                # Drain the message queue with error guard
                try:
                    has_msg, msg = win32gui.PeekMessage(None, 0, 0, PM_REMOVE)
                except win32gui.error:
                    # Window/message queue is in a bad state, exit gracefully
                    break

                if has_msg:
                    # msg is a tuple: (hwnd, message, wParam, lParam, time, pt)
                    if msg[1] == win32con.WM_QUIT:
                        break
                    win32gui.TranslateMessage(msg)
                    win32gui.DispatchMessage(msg)
                else:
                    # If the window has been destroyed, exit
                    if not win32gui.IsWindow(self.hwnd):
                        break
                    # Yield to the event loop
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Gracefully close on cancellation
            try:
                if win32gui.IsWindow(self.hwnd):
                    win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)
            except win32gui.error:
                pass
            raise

    def wndProc(self, hwnd, msg, wParam, lParam):
        if msg == win32con.WM_PAINT:
            hdc, paintStruct = win32gui.BeginPaint(hwnd)

            # Red outline, no fill
            pen = win32gui.CreatePen(win32con.PS_SOLID, 3, win32api.RGB(255, 0, 0))
            old_pen = win32gui.SelectObject(hdc, pen)
            null_brush = win32gui.GetStockObject(win32con.NULL_BRUSH)
            old_brush = win32gui.SelectObject(hdc, null_brush)

            # Draw all rectangles in the list
            for (x, y, w, h) in self.rects:
                win32gui.Rectangle(hdc, x, y, x + w, y + h)

            # Restore GDI objects and delete pen
            win32gui.SelectObject(hdc, old_pen)
            win32gui.SelectObject(hdc, old_brush)
            win32gui.DeleteObject(pen)

            win32gui.EndPaint(hwnd, paintStruct)
            return 0

        elif msg == win32con.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0

        return win32gui.DefWindowProc(hwnd, msg, wParam, lParam)
