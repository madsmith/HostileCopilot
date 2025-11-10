import pathlib
import sys
import pytest

# Ensure src is on sys.path for imports
ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hostile_copilot.client.components.locations.text import NormalizedName
from hostile_copilot.client.components.locations.locations import LocationProvider
from hostile_copilot.client.uexcorp import UEXCorpClient
from hostile_copilot.client.regolith import RegolithClient
from hostile_copilot.config import OmegaConfig, load_config

class TestNormalizedName:
    @pytest.mark.parametrize(
        "base, query",
        [
            ("The Moon XIX", "The Moon 19"),
            ("The Moon XIX", "moon nineteen"),
            ("The Moon XIX", "MOON-XIX"),
            ("ARC-L1", "ARC L1"),
            ("ARC-L1", "ARCL1"),
            ("ARC-L1", "arc-l-1"),
            ("ARC-L1", "ARC L 1"),
            ("Dudley & Daughters", "Dudley and Daughters"),
            ("Dudley & Daughters", "dudley & daughters"),
            ("Shubin Mining Facility SM0-10", "SM0-10"),
            ("Shubin Mining Facility SM0-10", "SM0 10"),
            ("Shubin Mining Facility SM0-10", "shubin sm010"),
            ("Pyro IV", "Pyro 4"),
            ("Pyro 4", "pyro IV"),
        ],
    )
    def test_matches_true(self, base: str, query: str):
        nn = NormalizedName(query)
        assert nn.matches(base) is True

    @pytest.mark.parametrize(
        "base, query",
        [
            ("The Moon XIX", "moon xviii"),
            ("The Moon XIX", "the-moon xix"),
            ("The Moon XIX", "sun 19"),
            ("ARC-L1", "ARC L2"),
            ("ARC-L1", "ARC-1L"),
            ("Dudley & Daughters", "Dudley or Daughters"),
            ("Shubin Mining Facility SM0-10", "SM0-11"),
            ("Shubin Mining Facility SM0-10", "SM1-10"),
            ("Pyro IV", "Pyro V"),
            ("Pyro IV", "Pyro 6"),
            ("Pyro IV", "pyro iv"),
        ],
    )
    def test_matches_false(self, base: str, query: str):
        nn = NormalizedName(query)
        assert nn.matches(base) is False

class TestLocationProvider:

    @pytest.fixture
    def config(self) -> OmegaConfig:
        return load_config()

    @pytest.fixture
    def uexcorp_client(self, config: OmegaConfig) -> UEXCorpClient:
        return UEXCorpClient(config)

    @pytest.fixture
    def regolith_client(self, config: OmegaConfig) -> RegolithClient:
        return RegolithClient(config)

    @pytest.fixture
    def provider(self, config: OmegaConfig, uexcorp_client: UEXCorpClient, regolith_client: RegolithClient) -> LocationProvider:
        return LocationProvider(config, uexcorp_client, regolith_client)
    
    @pytest.mark.asyncio
    async def test_search(self, provider: LocationProvider):
        results = await provider.search("Microtech")
        assert len(results) > 0
        assert len(results) == 2

        result = results[0]
        assert result.id == 'MIC'
        assert result.name == "MicroTech"
        assert 'Stanton IV' in result.aliases

        result = results[1]
        assert result.id == 87
        assert result.name == "Microtech Logistics Depot S4LD01"
        assert 'S4LD01' in result.aliases