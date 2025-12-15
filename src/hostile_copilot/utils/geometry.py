import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Union

Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]
AABB = tuple[float, float, float, float]


class BoundedRegion(ABC):
    """Abstract base class for regions with a finite 2D extent.

    Concrete subclasses must be able to provide a loose axis-aligned
    bounding box representation via ``to_aabb``.
    """

    @abstractmethod
    def to_aabb(self, pad: float = 0.0) -> AABB:
        """Return an axis-aligned bounding box (x1, y1, x2, y2).

        ``pad`` can be used by callers to expand the box uniformly.
        """
        raise NotImplementedError


class BoundingBox(BoundedRegion):
    """Axis-aligned bounding box defined by two corners.

    Parameters are (x1, y1, x2, y2) with no ordering constraints; the
    implementation normalizes them internally.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    def to_aabb(self, pad: float = 0.0) -> AABB:
        x1 = min(self.x1, self.x2) - pad
        y1 = min(self.y1, self.y2) - pad
        x2 = max(self.x1, self.x2) + pad
        y2 = max(self.y1, self.y2) + pad
        return x1, y1, x2, y2


class OrientedBoundingBox(BoundedRegion):
    """Oriented bounding box parameterized by center, size, and angle.

    Parameters are (xc, yc, w, h, angle_rad): center, width, height and
    rotation in radians (counter-clockwise).
    """

    def __init__(self, xc: float, yc: float, w: float, h: float, angle_rad: float) -> None:
        self.xc = float(xc)
        self.yc = float(yc)
        self.w = float(w)
        self.h = float(h)
        self.angle_rad = float(angle_rad)

    def __repr__(self) -> str:
        """Return a concise representation for debugging.

        Angle is shown in degrees for readability.
        """
        import math

        angle_deg = math.degrees(self.angle_rad)
        return (
            f"OrientedBoundingBox(xc={self.xc:.1f}, yc={self.yc:.1f}, "
            f"w={self.w:.1f}, h={self.h:.1f}, angle_deg={angle_deg:.2f})"
        )

    def corners(self) -> Corners:
        """Return the four corners of the OBB in image coordinates."""
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        local_pts = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        rotated = [
            rotate_point(px, py, self.angle_rad) for px, py in local_pts
        ]
        translated = [
            (self.xc + x, self.yc + y) for x, y in rotated
        ]
        return order_corners(translated)

    def to_aabb(self, pad: float = 0.0) -> AABB:
        # Delegate to get_bounding_box, then expand by pad.
        bb = get_bounding_box(self.corners())
        x1, y1, x2, y2 = bb.to_aabb(pad)
        return x1, y1, x2, y2


# Backwards-compatibility type for code that still uses raw tuples.
BoundingBoxLike = Union[BoundedRegion, AABB]

def rotate_point(px: float, py: float, angle_rad: float) -> Point:
    """
    Rotate point (px, py) by angle_rad (radians) counter-clockwise.
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return px * c - py * s, px * s + py * c

def order_corners(pts: list[Point]) -> Corners:
    """
    Order corners in counter-clockwise order.
    """
    p = np.array(pts, dtype=np.float32)
    s = p.sum(axis=1)
    diff = (p[:, 0] - p[:, 1])
    tl = p[np.argmin(s)]
    br = p[np.argmax(s)]
    tr = p[np.argmax(diff)]
    bl = p[np.argmin(diff)]
    return (
        (float(tl[0]), float(tl[1])),
        (float(tr[0]), float(tr[1])),
        (float(br[0]), float(br[1])),
        (float(bl[0]), float(bl[1])),
    )

def get_bounding_box(pts: Corners) -> BoundingBox:
    """Return a BoundingBox enclosing the given corners.

    The underlying coordinates are stored as an axis-aligned box, but
    callers can obtain the normalized (x1, y1, x2, y2) tuple via
    ``to_aabb()``.
    """
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))

def _to_aabb(box: BoundingBoxLike, pad: float = 0.0) -> AABB:
    """Convert any supported region representation into an AABB.

    Accepts both the new object-based API and legacy tuple forms:
    - BoundingBox / OrientedBoundingBox instances
    - (x1, y1, x2, y2)
    - (xc, yc, w, h, angle_rad)
    """

    if isinstance(box, BoundedRegion):
        return box.to_aabb(pad)

    # Legacy tuple forms
    if isinstance(box, tuple):
        if len(box) == 4:
            x1, y1, x2, y2 = box
            bb = BoundingBox(float(x1), float(y1), float(x2), float(y2))
            return bb.to_aabb(pad)
        if len(box) == 5:
            xc, yc, w, h, ang = box
            obb = OrientedBoundingBox(float(xc), float(yc), float(w), float(h), float(ang))
            return obb.to_aabb(pad)

    raise TypeError(f"Unsupported box type for _to_aabb: {type(box)!r}")


def is_overlapping(a: BoundingBoxLike, b: BoundingBoxLike, pad: float = 0.0) -> bool:
    """Loosely check if two bounded regions overlap.

    The check is performed on axis-aligned bounding boxes derived from the
    input regions. For oriented boxes this is intentionally approximate: we
    only test overlap between their AABBs.
    """

    ax1, ay1, ax2, ay2 = _to_aabb(a, pad)
    bx1, by1, bx2, by2 = _to_aabb(b, pad)

    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)