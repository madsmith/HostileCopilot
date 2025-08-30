from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Union, Literal, Generic, TypeVar, Callable, ClassVar

from hostile_copilot.client.uexcorp.types import BaseLocationID
from hostile_copilot.client.uexcorp.types import (
    PlanetID,
    MoonID,
    SpaceStationID,
    CityID,
    OrbitsID,
    OutpostID,
    PointOfInterestID,
)

TID = TypeVar("TID", bound=BaseLocationID)

class Location(BaseModel, Generic[TID]):
    id: TID
    name: str
    code: str | None = None

    # Subclasses should set this to their ID constructor, e.g., PlanetID
    id_factory: ClassVar[Callable[[int], TID] | None] = None

    @model_validator(mode="before")
    @classmethod
    def _wrap_id(cls, values: dict):
        raw_id = values.get("id")
        if isinstance(raw_id, int) and getattr(cls, "id_factory", None) is not None:
            values["id"] = cls.id_factory(raw_id)  # type: ignore[misc]
        return values

class Planet(Location[PlanetID]):
    kind: Literal["planet"] = "planet"
    id_factory = PlanetID

class Moon(Location[MoonID]):
    kind: Literal["moon"] = "moon"
    id_factory = MoonID

class SpaceStation(Location[SpaceStationID]):
    kind: Literal["space_station"] = "space_station"
    id_factory = SpaceStationID

    @model_validator(mode="before")
    @classmethod
    def derive_code(cls, values: dict):
        if "code" not in values and "nickname" in values:
            values["code"] = values["nickname"]
        return values

class City(Location[CityID]):
    kind: Literal["city"] = "city"
    id_factory = CityID

class Orbits(Location[OrbitsID]):
    kind: Literal["orbits"] = "orbits"
    id_factory = OrbitsID

class Outpost(Location[OutpostID]):
    kind: Literal["outpost"] = "outpost"
    id_factory = OutpostID

class PointOfInterest(Location[PointOfInterestID]):
    kind: Literal["point_of_interest"] = "point_of_interest"
    id_factory = PointOfInterestID

LocationType = Annotated[
    Union[Planet, Moon, SpaceStation, City, Orbits, Outpost, PointOfInterest],
    Field(discriminator="kind")
]
