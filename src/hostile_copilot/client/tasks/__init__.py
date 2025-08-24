from .base import Task
from .types import Coordinate
from .screen_bounding_box import GetScreenBoundingBoxTask
from .screen_get_location import GetScreenLocationTask
from .macro import MacroTask
from .extract_scan import MiningScanTask
from .mining_scan_grader import MiningScanGraderTask

__all__ = [
    "Task",
    "Coordinate",
    "GetScreenBoundingBoxTask",
    "GetScreenLocationTask",
    "MacroTask",
    "MiningScanTask",
    "MiningScanGraderTask"
]