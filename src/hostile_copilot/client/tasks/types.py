from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, Field

from hostile_copilot.mining_logger.types import ScanData

@dataclass
class Coordinate:
    x: int
    y: int


################################################################
# Mining Scanner Response Schema
################################################################


class ScanResponse(BaseModel):
    scan_data: Optional[ScanData] = Field(
        None,
        description="The parsed scan.  If no scan data is present in the image, return null."
    )
    