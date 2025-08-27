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
    
class CommodityData(BaseModel):
    name: str
    is_buyable: bool = Field(description="Whether the commodity can be bought in shops.")
    buy_price: int = Field(alias="price_buy", description="The price to buy the commodity in shops")
    is_sellable: bool = Field(description="Whether the commodity can be sold in shops.")
    sell_price: int = Field(alias="price_sell", description="The price to sell the commodity in shops")
    is_legal: bool = Field(description="Whether the commodity is considered legal trade.")
    is_mineral: bool = Field(description="Whether the commodity is a mineral.")
    is_refined: bool = Field(description="Whether the commodity is refined.")
    is_refinable: bool = Field(description="Whether the commodity can be refined into a refined version of the commodity.")