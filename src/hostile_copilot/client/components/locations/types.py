from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Annotated, Any, Callable, cast, ClassVar, Generic, get_args, Literal, TypeAlias, TypeVar, Union

from hostile_copilot.client.uexcorp.types import BaseLocationID
from hostile_copilot.client.uexcorp.types import (
    PlanetID,
    MoonID,
    SpaceStationID,
    CityID,
    OrbitsID,
    OutpostID,
    PointOfInterestID,
    GravityWellID,
    UEXType,
)

TID = TypeVar("TID", bound=BaseLocationID)

class Location(BaseModel, Generic[TID]):
    id: TID
    name: str
    code: str | None = None
    is_navigable: bool = False

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

WellType: TypeAlias = Literal["BELT", "CLUSTER", "LAGRANGE", "PLANET", "SATELLITE", "SYSTEM"]

WELLTYPE_SET = set(get_args(WellType))

def coerce_well_type(raw: Any) -> WellType | None:
    if not isinstance(raw, str):
        return None
    key = raw.strip().upper()
    if key in WELLTYPE_SET:
        return cast(WellType, key)
    return None

class GravityWell(BaseModel):
    kind: Literal["gravity_well"] = "gravity_well"
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
    is_navigable: bool = False

    @property
    def name(self) -> str:
        return self.label
    
    @field_validator("wellType", mode="before")
    @classmethod
    def _normalize_well_type(cls, v):
        return coerce_well_type(v)

LocationType = Annotated[
    Union[Planet, Moon, SpaceStation, City, Orbits, Outpost, PointOfInterest],
    Field(discriminator="kind")
]

LocationKind: TypeAlias = Literal[
    "planet",
    "moon",
    "space_station",
    "city",
    "orbits",
    "outpost",
    "point_of_interest",
    "gravity_well",
]

class LocationInfo(BaseModel):
    kind: LocationKind

    name: str
    aliases: list[str] | None = None

    id: BaseLocationID | None = None

    uex_id: BaseLocationID | None = None
    uex_type: UEXType | None = None

    code: str | None = None

    is_gravity_well: bool = False
    gravity_well_type: WellType | None = None

    is_navigable: bool = False
    is_space: bool | None = Field(default=None, repr=False)
    is_surface: bool | None = Field(default=None, repr=False)

    has_gems: bool | None = Field(default=None, repr=False)
    has_rocks: bool | None = Field(default=None, repr=False)

    parent_code: str | None = Field(default=None, repr=False)
    ancestor_codes: list[str] | None = Field(default=None, repr=False)

    raw_data: dict[str, Any] | None = Field(default=None, repr=False)

    def __str__(self):
        return f"{self.name} [{self.id} / {self.code}] ({self.kind})"

    @classmethod
    def from_planet(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("name")
        assert name is not None

        aliases = []
        name_origin = data.get("name_origin")
        if name_origin and name_origin != name:
            aliases.append(name_origin)

        uex_id = PlanetID(data['id'])

        return cls(
            kind="planet",
            name=name,
            aliases=aliases,
            id=uex_id,
            uex_id=uex_id,
            uex_type="planet",
            code=data.get("code"),
            is_gravity_well=False,
            gravity_well_type=None,
            raw_data=data
        )

    @classmethod
    def from_moon(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("name")
        assert name is not None

        aliases = []
        name_origin = data.get("name_origin")
        if name_origin and name_origin != name:
            aliases.append(name_origin)

        uex_id = MoonID(data['id'])

        return cls(
            kind="moon",
            name=name,
            aliases=aliases,
            id=uex_id,
            uex_id=uex_id,
            uex_type="moon",
            code=data.get("code"),
            is_gravity_well=False,
            gravity_well_type=None,
            raw_data=data
        )

    @classmethod
    def from_space_station(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("name")
        assert name is not None

        aliases = []
        nickname = data.get("nickname")
        if nickname and nickname != name:
            aliases.append(nickname)

        uex_id = SpaceStationID(data['id'])

        return cls(
            kind="space_station",
            name=name,
            aliases=aliases,
            id=uex_id,
            uex_id=uex_id,
            uex_type="space_station",
            code=data.get("code"),
            is_gravity_well=False,
            gravity_well_type=None,
            raw_data=data
        )

    @classmethod
    def from_city(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("name")
        assert name is not None

        uex_id = CityID(data['id'])

        return cls(
            kind="city",
            name=name,
            id=uex_id,
            uex_id=uex_id,
            uex_type="city",
            code=data.get("code"),
            is_gravity_well=False,
            gravity_well_type=None,
            raw_data=data
        )

    @classmethod
    def from_outpost(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("name")
        assert name is not None

        aliases = []
        nickname = data.get("nickname")
        if nickname and nickname != name:
            aliases.append(nickname)

        uex_id = OutpostID(data['id'])

        return cls(
            kind="outpost",
            name=name,
            aliases=aliases,
            id=uex_id,
            uex_id=uex_id,
            uex_type="outpost",
            code=data.get("code"),
            is_gravity_well=False,
            gravity_well_type=None,
            raw_data=data
        )

    @classmethod
    def from_point_of_interest(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("name")
        assert name is not None

        aliases = []
        nickname = data.get("nickname")
        if nickname and nickname != name:
            aliases.append(nickname)

        uex_id = PointOfInterestID(data['id'])

        return cls(
            kind="point_of_interest",
            name=name,
            aliases=aliases,
            id=uex_id,
            uex_id=uex_id,
            uex_type="point_of_interest",
            code=data.get("code"),
            is_gravity_well=False,
            gravity_well_type=None,
            raw_data=data
        )

    @classmethod
    def from_gravity_well(cls, data: dict[str, Any]) -> "LocationInfo":
        name = data.get("label")
        assert name is not None


        regolith_id = GravityWellID(data['id'])

        return cls(
            kind="gravity_well",
            name=name,
            wellType=data.get("wellType"),
            id=regolith_id,
            uex_type=cls._regolith_type_to_uex_type(data.get("wellType")),
            code=data.get("id"),
            is_gravity_well=True,
            gravity_well_type=data.get("wellType"),
            has_gems=data.get("hasGems"),
            has_rocks=data.get("hasRocks"),
            is_space=data.get("isSpace"),
            is_surface=data.get("isSurface"),
            raw_data=data
        )

    @classmethod
    def _regolith_type_to_uex_type(cls, v: WellType | None) -> UEXType | None:
        # ["BELT", "CLUSTER", "LAGRANGE", "PLANET", "SATELLITE", "SYSTEM"]
        if v == "PLANET":
            return "planet"

        elif v == "SATELLITE":
            return "moon"

        elif v == "SYSTEM":
            return "star_system"

        elif v == "LAGRANGE":
            return "orbits"
        
        return None