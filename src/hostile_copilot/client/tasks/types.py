from dataclasses import dataclass
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator

@dataclass
class Coordinate:
    x: int
    y: int


################################################################
# Mining Scanner Response Schema
################################################################

class ScanItem(BaseModel):
    material: str = Field(
        description="The name of the material"
    )
    percentage: float = Field(
        description="The percentage of the material"
    )

    @field_validator("percentage", mode="before")
    @classmethod
    def normalize_percentage(cls, v):
        if isinstance(v, str):
            v = v.strip().replace("%", "")
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"Invalid percentage: {v}")

DifficultyT = Literal["Easy", "Medium", "Hard", "Impossible", "Unknown"]

class ScanData(BaseModel):
    object_type: str = Field(
        description="The type of object scanned. e.g. Asteroid, Sal"
    )
    object_subtype: str | None = Field(
        description="The subtype of the object scanned.  Usually found within parentheses.. e.g. Q-Type"
    )
    mass: int = Field(
        description="The mass of the object"
    )
    resistance: int = Field(
        description="The resistance of the object as a percentage"
    )
    instability: float = Field(
        description="The instability of the object"
    )
    difficulty: Optional[DifficultyT] = Field(
        description="The difficulty rating from Easy to Impossible.  Not always present in scan."
    )
    size: float = Field(
        description="The size of the object measured in SCU"
    )
    composition: list[ScanItem] = Field(
        description="Mapping of the material name to percentage composition found in the scan results",
        min_items=1
    )

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if isinstance(v, str):
            val = v.strip().lower()
            if val in {"unknown", "unk", "n/a", "none"}:
                return "Unknown"
            if val == "easy":
                return "Easy"
            if val == "medium":
                return "Medium"
            if val == "hard":
                return "Hard"
            if val == "impossible":
                return "Impossible"
        return v

class ScanResponse(BaseModel):
    scan_data: Optional[ScanData] = Field(
        None,
        description="The parsed scan.  If no scan data is present in the image, return null."
    )
    