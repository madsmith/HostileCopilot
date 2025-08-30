import json
import logging
from pathlib import Path
import requests
from typing import Any
import time
from urllib.parse import urljoin

from hostile_copilot.config import OmegaConfig

from .types import StarSystemID

logger = logging.getLogger(__name__)

class UEXCorpException(Exception):
    pass

class UEXCorpAPIException(Exception):
    def __init__(self, http_code: int, message: str):
        self.http_code = http_code
        self.message = message

class UEXCorpClient:
    def __init__(self, config: OmegaConfig):
        self._config: OmegaConfig = config

        self._bearer_token: str = config.get("uexcorp.bearer_token")
        self._file_cache_path: Path = Path(config.get("uexcorp.file_cache_path", "cache/uexcorp"))
        self._cache_expiry: int = config.get("uexcorp.cache_expiry", 24 * 3600)

        assert self._bearer_token is not None, "Missing config 'uexcorp.bearer_token'"

    async def api_call(self, endpoint: str, method: str = "GET", data: dict | None = None):
        """
        Make an API call to UEXCorp.

        Args:
            endpoint (str): The endpoint to call.
            method (str, optional): The HTTP method to use. Defaults to "GET".
            data (dict | None, optional): The data to send with the request. Defaults to None.

        Returns:
            dict: The response from the API, including the http_code and message.
        """ 
        base_uri = self._config.get("uexcorp.api_url", "https://api.uexcorp.space/2.0/")
        url = urljoin(base_uri, endpoint)
        headers = {
            "Authorization": f"Bearer {self._bearer_token}",
            "Content-Type": "application/json",
        }
        
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported request method: {method}")

        if response.status_code != 200:
            try:
                data = response.json()
                raise UEXCorpException(f"Request failed: {data['http_code']} - {data}")
            except ValueError:
                raise UEXCorpException(f"Request failed: {response.status_code} - {response.text}")

        try:
            return response.json()
        except ValueError:
            raise UEXCorpException(f"Request failed with status code {response.status_code}")
    
    async def api_call_cached_data(self, endpoint: str, method: str = "GET", data: dict | None = None):
        """
        Make an API call to UEXCorp and cache the response.

        Args:
            endpoint (str): The endpoint to call.
            method (str, optional): The HTTP method to use. Defaults to "GET".
            data (dict | None, optional): The data to send with the request. Defaults to None.

        Returns:
            dict: The data response from the API call
        """ 

        cache_filename = self._get_cache_filename(endpoint, method, data)
        cache_path = self._file_cache_path / cache_filename

        if await self.is_cached(endpoint, cache_path):
            try:
                with open(cache_path, "r") as f:
                    return json.load(f) 
            except Exception as e:
                logger.exception(f"Failed to load cache: {e}")
                try:
                    cache_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.exception(f"Failed to remove cache: {e}")
                pass
        
        response = await self.api_call(endpoint, method, data)

        if response is None:
            raise UEXCorpException("No response from API")
        
        
        if response["http_code"] != 200:
            raise UEXCorpAPIException(response['http_code'], response['messsage'])

        if "data" not in response:
            logger.warning(f"No data in response: {response}")
            raise UEXCorpAPIException(500, "No data in response")

        response_data = response["data"]

        await self._cache_data(cache_path, response_data)
        
        return response_data

    async def _cache_data(self, cache_path: Path, data: dict[str, Any]):
        """
        Cache the response data.

        Args:
            cache_path (Path): The path to the cache file.
            data (dict[str, Any]): The data to cache.
        """
        # ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "w") as f:
            try:
                json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
                pass

    async def is_cached(self, endpoint: str, cache_path: Path) -> bool:
        if not cache_path.exists():
            return False

        cache_expiry = self._config.get(f"uexcorp.api_calls.{endpoint}.cache_expiry", self._cache_expiry)
        
        return (time.time() - cache_path.stat().st_mtime) < cache_expiry

    def _get_cache_filename(self, endpoint: str, method: str, data: dict | None) -> str | None:
        if method != "GET":
            # Only GET requests are cached
            return None

        if not data:
            return f"{endpoint}.json"

        # Build deterministic name like: endpoint_key1_val1_key2_val2.json
        parts: list[str] = [endpoint]
        for key in sorted(data.keys()):
            val = data[key]
            if isinstance(val, (list, tuple, dict)):
                return None
            v_str = str(val)

            # Make filesystem-safe-ish: replace spaces
            v_str = v_str.replace(" ", "_")
            parts.append(f"{key}={v_str}")

        return f"{'_'.join(parts)}.json"

    #################################################################
    # API Calls
    #################################################################

    async def fetch_commodities(self) -> list[dict[str, Any]]:
        """
        Fetch a list of commodities from UEXCorp.

        Returns:
            list[dict[str, Any]]: A list of commodities.
        """
        return await self.api_call_cached_data("commodities")

    async def fetch_star_systems(self) -> list[dict[str, Any]]:
        """
        Fetch a list of star systems from UEXCorp.

        Returns:
            list[dict[str, Any]]: A list of star systems.
        """
        return await self.api_call_cached_data("star_systems")
    
    async def fetch_moons(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of moons from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch moons for.

        Returns:
            list[dict[str, Any]]: A list of moons.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("moons", "GET", {"id_star_system": star_system})

    async def fetch_planets(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of planets from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch planets for.

        Returns:
            list[dict[str, Any]]: A list of planets.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("planets", "GET", {"id_star_system": star_system})

    async def fetch_orbits(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of orbits from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch orbits for.

        Returns:
            list[dict[str, Any]]: A list of orbits.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("orbits", "GET", {"id_star_system": star_system})

    async def fetch_stations(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of stations from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch stations for.

        Returns:
            list[dict[str, Any]]: A list of stations.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("space_stations", "GET", {"id_star_system": star_system})

    async def fetch_cities(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of cities from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch cities for.

        Returns:
            list[dict[str, Any]]: A list of cities.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("cities", "GET", {"id_star_system": star_system})

    async def fetch_outposts(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of outposts from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch outposts for.

        Returns:
            list[dict[str, Any]]: A list of outposts.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("outposts", "GET", {"id_star_system": star_system})

    async def fetch_points_of_interest(self, star_system: StarSystemID | int) -> list[dict[str, Any]]:
        """
        Fetch a list of points of interest from UEXCorp.

        Args:
            star_system (StarSystemID | int): The ID of the star system to fetch points of interest for.

        Returns:
            list[dict[str, Any]]: A list of POIs.
        """
        assert isinstance(star_system, (StarSystemID, int))

        if isinstance(star_system, StarSystemID):
            star_system = star_system.value

        return await self.api_call_cached_data("poi", "GET", {"id_star_system": star_system})
        