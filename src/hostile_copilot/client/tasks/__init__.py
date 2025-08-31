from .base import Task
from .types import Coordinate, SetLocationResponse
from .screen_bounding_box import GetScreenBoundingBoxTask
from .screen_get_location import GetScreenLocationTask
from .macro import MacroTask
from .extract_scan import MiningScanTask
from .mining_scan_grader import MiningScanGraderTask
from .nav_set_route import NavSetRouteTask

__all__ = [
    "Task",
    "Coordinate",
    "GetScreenBoundingBoxTask",
    "GetScreenLocationTask",
    "MacroTask",
    "MiningScanTask",
    "MiningScanGraderTask",
    "NavSetRouteTask",
    "SetLocationResponse"
]