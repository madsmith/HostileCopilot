from dataclasses import dataclass
from difflib import get_close_matches
import numpy as np
import re
from typing import Any
from sklearn.cluster import KMeans

from hostile_copilot.config import OmegaConfig
from hostile_copilot.mining_logger import ScanData
from hostile_copilot.client.uexcorp import UEXCorpClient

from hostile_copilot.utils.logging import get_trace_logger
logger = get_trace_logger(__name__)

COMMODITY_NAME_REGEXP = re.compile(r"^(.*?)(?:\s(?:\((?:raw|ore)\)|ore))?$", re.IGNORECASE)

@dataclass
class CommodityGrade:
    commodity: str
    price: float
    value: float
    size: float
    tier: int

@dataclass
class ScanGrade:
    total_value: float
    best_tier: int
    commodity_grades: list[CommodityGrade]

class CommodityGrader:
    def __init__(self, config: OmegaConfig):
        self._config = config
        self._commodities: list[dict[str, Any]] = []
        self._tier_map: dict[str, int] = {}
        self._tier_predictor: KMeans | None = None

    async def initialize(self, uexcorp_client: UEXCorpClient):
        self._commodities = await uexcorp_client.fetch_commodities()
        self._initialize_tier_map()

    def grade_scan(self, scan_data: ScanData) -> ScanGrade:
        scan_grade = ScanGrade(
            total_value=0,
            best_tier=0,
            commodity_grades=[]
        )

        total_value = 0
        for scan_item in scan_data.composition:
            material = self.identify_commodity(scan_item.material)
            tier = self.get_tier(material)
            price = self._get_price(material)
            logger.debug(f"Percentage: {scan_item.percentage} | Size: {scan_data.size}")
            size = round(scan_item.percentage * scan_data.size / 100, 1)
            value = round(price * size)
            logger.debug(f"Grading {material}: {price} * {size} = {value}")

            total_value += value

            scan_grade.commodity_grades.append(
                CommodityGrade(
                    commodity=scan_item.material,
                    price=price,
                    value=value,
                    size=size,
                    tier=tier
                )
            )

        scan_grade.total_value = round(total_value)
        scan_grade.best_tier = min(scan_grade.commodity_grades, key=lambda x: x.tier).tier

        return scan_grade

    def identify_commodity(self, commodity_name: str) -> str:
        logger.trace(f"Identifying commodity: {commodity_name}")
        refined_commodity = self.refineable_to_refined(commodity_name)

        known_commodities = [commodity["name"].title() for commodity in self._commodities]

        search_commodity = refined_commodity.title()
        close_matches = get_close_matches(search_commodity, known_commodities, cutoff=0.8)
        if close_matches:
            logger.trace(f"Close matches: {close_matches}")
            return close_matches[0]

        logger.debug(f"Could not identify commodity: {refined_commodity}")
        return f"Unknown [{refined_commodity}]"

    def get_tier(self, commodity: str) -> int:
        assert self._tier_predictor is not None, "Tier predictor is not initialized"

        # Get price of commodity
        price = self._get_price(commodity)

        data = np.array([[price]])
        prediction = self._tier_predictor.predict(data)

        tier_label = self._tier_map.get(prediction[0], 0)
        return tier_label
    
    def refineable_to_refined(self, name: str) -> str | None:
        mapping = self._config.get("refineable_mapping", {})

        if name in mapping:
            return mapping[name]

        match = COMMODITY_NAME_REGEXP.match(name)
        if not match:
            logger.warning(f"Could not identify commodity: {name}")
            return None
        
        return match.group(1)

    def _initialize_tier_map(self):
        tier_count = self._config.get("commodity_grader.tier_count", 7)

        # Initialize K-Means model
        kmeans = KMeans(n_clusters=tier_count, random_state=42, n_init=10)

        # Filter refined commodities
        refined_commodities = [commodity for commodity in self._commodities if commodity["is_refined"]]

        prices = np.array([
            [int(commodity["price_sell"]),]
            for commodity in refined_commodities
        ])

        clusters = kmeans.fit_predict(prices)

        # Sort cluster labels by price descending
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = np.argsort(cluster_centers[:, 0])[::-1]

        # Assign tiers to cluster labels
        self._tier_map = {cluster_labels[i]: i + 1 for i in range(tier_count)}
        self._tier_predictor = kmeans

        # Build tier map for debugging
        # "tiermap.json" is a JSON file with the following format:
        # {
        #   "Tier 1": [
        #     { name: "commodity_name_1", price: 123456 },
        #     { name: "commodity_name_2", price: 123456 },
        #     ...
        #   ]
        #   ...
        # }

    def save_tier_map(self, path: str | None = None):
        if path is None:
            path = "tiermap.json"

        refined_commodities = [commodity for commodity in self._commodities if commodity["is_refined"]]

        with open(path, "w") as f:
            data = {}
            for key, value in self._tier_map.items():
                data[f"Tier {value}"] = [
                    {
                        "name": commodity["name"],
                        "price": commodity["price_sell"]
                    }
                    for commodity in refined_commodities
                    if self.get_tier(commodity["name"]) == value
                ]
            import json
            json.dump(data, f, indent=2)

    def _get_price(self, commodity: str) -> int:
        refined_commodity = self.refineable_to_refined(commodity)
        
        lookup_commodity = refined_commodity if refined_commodity else commodity
        logger.trace(f"Input Commodity: {commodity}")
        logger.trace(f"Refined Commodity: {refined_commodity}")
        logger.trace(f"Lookup Commodity: {lookup_commodity}")
        # logger.debug(self._commodities)

        for commodity in self._commodities:
            if commodity["name"].lower() == lookup_commodity.lower():
                return int(commodity["price_sell"])
        
        return 0