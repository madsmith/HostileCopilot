from dataclasses import dataclass
from typing import Optional, Union
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

class PingSignatureReadout(BaseModel):
    infrared: int = Field(description="The infrared ping value")
    em: int = Field(description="The EM ping value")
    crosssection: int = Field(description="The crosssection ping value")

class PingResourceSummary(BaseModel):
    radar: int = Field(description="The sum value of the resources found at the ping.")

PingData = Union[PingResourceSummary, PingSignatureReadout]

class PingResponse(BaseModel):
    ping_data: Optional[PingData] = Field(
        None,
        description="The parsed ping.  If no ping data is present in the image, return null."
    )

class NavSetRouteResponse(BaseModel):
    success: bool
    message: str
    
class SetLocationResponse(BaseModel):
    success: bool
    message: str

class CommodityData(BaseModel):
    name: str
    is_buyable: bool = Field(description="Whether the commodity can be bought in shops.")
    buy_price: int = Field(alias="price_buy", description="The price to buy the commodity in shops")
    is_sellable: bool = Field(description="Whether the commodity can be sold in shops.")
    sell_price: int = Field(alias="price_sell", description="The price to sell the commodity in shops")
    is_illegal: bool = Field(description="Whether the commodity is considered illegal trade.")
    is_mineral: bool = Field(description="Whether the commodity is a mineral.")
    is_refined: bool = Field(description="Whether the commodity is refined.")
    is_refinable: bool = Field(description="Whether the commodity can be refined into a refined version of the commodity.")