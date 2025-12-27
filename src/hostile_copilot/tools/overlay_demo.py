from __future__ import annotations

import argparse
import sys
import threading
from typing import Callable, Dict, Tuple

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import QApplication

from hostile_copilot.client.components.ui import Overlay
from hostile_copilot.client.components.ui.components import Box, LabeledBox, TextBox, Drawable


# -------------------------
# Utilities
# -------------------------

COLOR_MAP: Dict[str, Tuple[int, int, int, int]] = {
    "red": (255, 0, 0, 255),
    "green": (0, 255, 0, 255),
    "blue": (0, 0, 255, 255),
    "yellow": (255, 255, 0, 255),
    "purple": (128, 0, 128, 255),
    "cyan": (0, 255, 255, 255),
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
}


def parse_color(name: str) -> QColor:
    if name.lower() in COLOR_MAP:
        r, g, b, a = COLOR_MAP[name.lower()]
        return QColor(r, g, b, a)
    # Fallback: try QColor name directly
    c = QColor(name)
    if c.isValid():
        return c
    # Default to white if unknown
    return QColor(255, 255, 255, 255)


# -------------------------
# Command handling
# -------------------------

class CommandBus(QObject):
    command_received = Signal(str)


def tokenize(line: str) -> list[str]:
    # Simple tokenizer that respects quoted strings
    tokens: list[str] = []
    cur = []
    in_quotes = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if not in_quotes and ch.isspace():
            if cur:
                tokens.append(''.join(cur))
                cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if cur:
        tokens.append(''.join(cur))
    return tokens


def handle_command(overlay: Overlay, tokens: list[str]) -> str:
    if not tokens:
        return ""
    cmd = tokens[0].lower()

    try:
        if cmd in ("quit", "exit"):
            QApplication.quit()
            return "Exiting"

        if cmd == "help":
            return (
                "Commands:\n"
                "  help                         - show this help\n"
                "  clear                        - clear all drawables\n"
                "  box x1 y1 x2 y2 <color>      - draw Box\n"
                "  lbox x1 y1 x2 y2 \"text\" <color> - draw LabeledBox\n"
                "  tbox x y \"text\" <anchor> - draw TextBox (anchor: top_left|top_center|center)\n"
                "  quit | exit                  - exit app\n"
            )

        if cmd == "clear":
            overlay.clear_drawables()
            return "Cleared drawables"

        if cmd == "box":
            # box x1 y1 x2 y2 color
            if len(tokens) < 6:
                return "Usage: box x1 y1 x2 y2 <color>"
            x1, y1, x2, y2 = map(int, tokens[1:5])
            color = parse_color(tokens[5])
            overlay.add_drawable(Box(x1=x1, y1=y1, x2=x2, y2=y2, color=color))
            return f"Box added ({x1},{y1})-({x2},{y2}) {tokens[5]}"

        if cmd == "lbox":
            # lbox x1 y1 x2 y2 "label" color
            if len(tokens) < 7:
                return "Usage: lbox x1 y1 x2 y2 \"label\" <color>"
            x1, y1, x2, y2 = map(int, tokens[1:5])
            label = tokens[5]
            color = parse_color(tokens[6])
            overlay.add_drawable(LabeledBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label, color=color))
            return f"LabeledBox added ({x1},{y1})-({x2},{y2}) '{label}' {tokens[6]}"

        if cmd == "tbox":
            # tbox x y "text" anchor
            if len(tokens) < 5:
                return "Usage: tbox x y \"text\" <anchor>"
            x, y = int(tokens[1]), int(tokens[2])
            text = tokens[3]
            anchor = tokens[4]
            overlay.add_drawable(TextBox(x=x, y=y, text=text, anchor=anchor))
            return f"TextBox added ({x},{y}) '{text}' {anchor}"

        return f"Unknown command: {cmd}. Type 'help' for options."
    except Exception as e:
        return f"Error: {e}"


# -------------------------
# Main app
# -------------------------

def repl_thread(bus: CommandBus, done_event: threading.Event):
    print("Overlay demo REPL ready. Type 'help' for commands. Ctrl+C to exit.")
    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            bus.command_received.emit(line.strip())
            # Block next prompt until UI thread processed the command
            done_event.wait()
            done_event.clear()
    finally:
        # If stdin closes, quit the app
        bus.command_received.emit("exit")


def main():
    parser = argparse.ArgumentParser(description="Overlay drawing demo with REPL")
    parser.add_argument("--monitor", "--mon", "-m", type=int, default=None,
                        help="Which monitor to show the overlay on (1-based index). Defaults to primary.")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    screens = QGuiApplication.screens()
    screen = screens[(args.monitor - 1)] if (args.monitor and 1 <= args.monitor <= len(screens)) else screens[0]

    overlay = Overlay()
    overlay.showOnScreen(screen)

    bus = CommandBus()
    done_event = threading.Event()

    def on_command(line: str):
        try:
            tokens = tokenize(line)
            response = handle_command(overlay, tokens)
            if response:
                print(response)
        except Exception as e:
            # Defensive: ensure REPL isn't blocked even if unexpected errors occur
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                pass
            print(f"Handler error: {e}")
        finally:
            # Signal the REPL that handling is complete regardless of success/failure
            done_event.set()

    bus.command_received.connect(on_command)

    t = threading.Thread(target=repl_thread, args=(bus, done_event), daemon=True)
    t.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
