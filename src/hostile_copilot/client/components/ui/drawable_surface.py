from __future__ import annotations

from typing import Sequence, Protocol, runtime_checkable

from PySide6.QtGui import QScreen

from .components import Drawable


@runtime_checkable
class DrawableSurface(Protocol):
    """Structural interface for surfaces that can display Drawable items on a given screen.
    Implementing classes do NOT need to inherit from this; methods are checked structurally.
    """

    def showOnScreen(self, screen: QScreen) -> None:  # noqa: N802 (match existing naming convention)
        """Show the surface on the specified QScreen."""
        ...

    def clear_drawables(self) -> None:
        """Remove all drawables and trigger a repaint."""
        ...

    def add_drawable(self, drawable: Drawable) -> None:
        """Add a single drawable and trigger a repaint."""
        ...

    def add_drawables(self, drawables: Sequence[Drawable]) -> None:
        """Add multiple drawables and trigger a repaint."""
        ...

    def set_drawables(self, drawables: Sequence[Drawable]) -> None:
        """Replace all drawables with the provided sequence and trigger a repaint."""
        ...

    def screen_width(self) -> int:
        """Return the physical screen width in pixels for the surface's target screen."""
        ...

    def screen_height(self) -> int:
        """Return the physical screen height in pixels for the surface's target screen."""
        ...
