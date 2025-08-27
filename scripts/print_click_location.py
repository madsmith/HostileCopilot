from __future__ import annotations

# Simple helper to print the screen coordinates of the next left-click.
# Press Esc to exit without selecting.

from pynput import mouse, keyboard
import sys

_mouse_listener: mouse.Listener | None = None


def on_click(x: int, y: int, button: mouse.Button, pressed: bool):
    if pressed and button == mouse.Button.left:
        print(f"{x},{y}")
        sys.stdout.flush()
        # Stop after first left-click
        return False
    return None


def on_key_press(key):
    # Allow exiting with Esc
    global _mouse_listener
    if key == keyboard.Key.esc:
        print("Esc pressed. Exiting.")
        if _mouse_listener is not None:
            _mouse_listener.stop()
        # Stop keyboard listener
        return False
    return None


def main():
    print("Click anywhere (left button) to print coordinates, or press Esc to exit.")
    kb_listener = keyboard.Listener(on_press=on_key_press)
    kb_listener.start()

    global _mouse_listener
    _mouse_listener = mouse.Listener(on_click=on_click)
    _mouse_listener.start()

    # Wait for mouse listener to finish (either click or Esc)
    _mouse_listener.join()
    # Ensure keyboard listener stops
    if kb_listener.running:
        kb_listener.stop()


if __name__ == "__main__":
    main()
