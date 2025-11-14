from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

@dataclass(frozen=True, slots=True)
class BaseLocationID:
    value: int | str
    namespace: str

    def __str__(self) -> str:
        return f"{self.namespace}:{self.value}"

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        return f"{classname}({self.value})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BaseLocationID):
            return self.value == other.value and self.namespace == other.namespace
        elif isinstance(other, str):
            return str(self) == other or self.value == other
        else:
            return self.value == other

class StarSystemID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "star_system")

class PlanetID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "planet")

class MoonID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "moon")

class SpaceStationID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "space_station")

class CityID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "city")

class OutpostID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "outpost")

class PointOfInterestID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "point_of_interest")

class OrbitID(BaseLocationID):
    def __init__(self, value: int):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "orbits")

class GravityWellID(BaseLocationID):
    def __init__(self, value: str):
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "namespace", "gravity_well")

UEXType: TypeAlias = Literal[
    "star_system",
    "planet",
    "moon",
    "space_station",
    "city",
    "outpost",
    "point_of_interest",
    "orbits",
]
