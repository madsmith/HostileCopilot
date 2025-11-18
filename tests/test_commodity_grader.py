import pathlib
import sys
import pytest

# Ensure src is on sys.path for imports
ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hostile_copilot.config import OmegaConfig, load_config
from hostile_copilot.client.commodity_grader import CommodityGrader
from hostile_copilot.mining_logger.types import ScanData, ScanItem
from hostile_copilot.client.uexcorp import UEXCorpClient


class TestCommodityGrader:
    @pytest.fixture
    def config(self) -> OmegaConfig:
        return load_config()

    @pytest.fixture
    def uexcorp_client(self, config: OmegaConfig) -> UEXCorpClient:
        return UEXCorpClient(config)

    @pytest.mark.parametrize(
        "scan_name, expected",
        [
            ("Quantainium", "Quantainium"),
            ("Quantainium (Raw)", "Quantainium"),
            ("Quantainium (Ore)", "Quantainium"),

            ("QUANTAINIUM", "Quantainium"),
            ("QUANTAINIUM (RAW)", "Quantainium"),
            ("QUANTAINIUM (ORE)", "Quantainium"),
            
            ("Quantanium", "Quantainium"),
            ("Quantanium (Raw)", "Quantainium"),
            ("Quantanium (Ore)", "Quantainium"),

            ("QUANTANIUM", "Quantainium"),
            ("QUANTANIUM (RAW)", "Quantainium"),
            ("QUANTANIUM (ORE)", "Quantainium"),

            ("Bexalite (Raw)", "Bexalite"),
            
        ]
    )
    @pytest.mark.asyncio
    async def test_identify_commodity(
        self,
        config: OmegaConfig,
        uexcorp_client: UEXCorpClient,
        scan_name: str,
        expected: str,
    ):
        grader = CommodityGrader(config)

        await grader.initialize(uexcorp_client)

        assert grader.identify_commodity(scan_name) == expected


    @pytest.mark.asyncio
    async def test_grade_scan_computes_values_and_tiers(self, config: OmegaConfig, uexcorp_client: UEXCorpClient):
        grader = CommodityGrader(config)

        await grader.initialize(uexcorp_client)

        # Sample scan data provided by user
        scan = ScanData(
            object_type="Asteroid",
            object_subtype="M-Type",
            mass=7358,
            resistance=36,
            instability=65.28,
            difficulty="Unknown",
            size=28.66,
            composition=[
                ScanItem(material="QUANTANIUM", percentage=33.19),
                ScanItem(material="TARANITE", percentage=45.37),
                ScanItem(material="ALUMINUM", percentage=21.42),
                ScanItem(material="INERT MATERIALS", percentage=11.48),
            ],
        )

        result = grader.grade_scan(scan)

        # Expected sizes (rounded to 1 decimal place inside grade_scan)
        # 10.52 * 33.19% = 3.491588 -> 3.5
        # 10.52 * 45.37% = 4.772924 -> 4.8
        # 10.52 * 21.42% = 2.253384 -> 2.3
        expected_sizes = {
            "Agricium (Ore)": 3.5,
            "Quartz (Raw)": 4.8,
            "Inert Materials": 2.3,
        }

        # Expected values using our test prices (rounded to int inside grade_scan)
        # Agricium: 1000 * 3.5 = 3500
        # Quartz: 100 * 4.8 = 480
        # Inert: 10 * 2.3 = 23
        expected_values = {
            "Agricium (Ore)": 3500,
            "Quartz (Raw)": 480,
            "Inert Materials": 23,
        }

        # Total value is rounded at the end
        assert result.total_value == sum(expected_values.values())

        # Collect graded commodities by their scan material names
        graded = {c.commodity: c for c in result.commodity_grades}

        for name, size in expected_sizes.items():
            assert name in graded
            assert graded[name].size == pytest.approx(size, rel=0, abs=1e-9)

        for name, value in expected_values.items():
            assert graded[name].value == value

        # Best tier should be the smallest tier label (1 is highest value cluster)
        min_tier = min(c.tier for c in result.commodity_grades)
        assert result.best_tier == min_tier == 1

    @pytest.mark.parametrize(
        "commodity_name, min_price",
        [
            ("Quantainium", 15000),
            ("Taranite", 6000),
            ("Bexalite", 6000),
            ("Aluminum", 200),
        ]
    )
    @pytest.mark.asyncio
    async def test_get_price_minimum(
        self,
        config: OmegaConfig,
        uexcorp_client: UEXCorpClient,
        commodity_name: str,
        min_price: int
    ):
        grader = CommodityGrader(config)

        await grader.initialize(uexcorp_client)

        result = grader._get_price(commodity_name)

        assert result >= min_price