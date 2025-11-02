from typing import Literal, TypeAlias
from pydantic import BaseModel

WellType: TypeAlias = Literal["BELT", "CLUSTER", "LAGRANGE", "PLANET", "SATELLITE", "SYSTEM"]

class GravityWell(BaseModel):
    label: str
    wellType: WellType
    id: str
    system: str
    depth: int
    parent: str | None = None
    parents: list[str] | None = None
    parentType: WellType | None = None
    isSpace: bool | None = None
    isSurface: bool | None = None
    hasGems: bool | None = None
    hasRocks: bool | None = None
