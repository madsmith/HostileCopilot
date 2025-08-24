from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

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
    
    @field_validator("resistance", mode="before")
    @classmethod
    def parse_resistance(cls, v):
        if isinstance(v, str):
            v = v.strip().replace("%", "")
        try:
            return int(v)
        except Exception:
            return 0

    @field_validator("object_type", "object_subtype", mode="before")
    def normalize_names(cls, v: str) -> str:
        if not v:
            return "Unknown"
        return v.strip().title()

    @field_validator("mass", "resistance", mode="before")
    def ensure_positive_int(cls, v):
        try:
            v = int(v)
        except Exception:
            return 0
        return max(0, v)

    @field_validator("instability", "size", mode="before")
    def ensure_nonnegative_float(cls, v):
        try:
            v = float(v)
        except Exception:
            return 0.0
        return max(0.0, v)

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