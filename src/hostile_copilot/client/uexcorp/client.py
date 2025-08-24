import json
import logging
from pathlib import Path
import requests

from hostile_copilot.config import OmegaConfig

logger = logging.getLogger(__name__)

class UEXCorpException(Exception):
    pass

class UEXCorpClient:
    def __init__(self, config: OmegaConfig):
        self._config: OmegaConfig = config

        self._bearer_token: str = config.get("uexcorp.bearer_token")
        self._file_cache_path: Path = Path(config.get("uexcorp.file_cache_path", "cache/uexcorp"))
        self._cache_expiry: int = config.get("uexcorp.cache_expiry", 24 * 3600)

        assert self._bearer_token is not None, "Missing config 'uexcorp.bearer_token'"

    async def api_call(self, endpoint: str, method: str = "GET", data: dict | None = None):
        url = "https://api.uexcorp.com/2.0/" + endpoint
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
            raise UEXCorpException(f"Request failed with status code {response.status_code}")

        try:
            return response.json()
        except ValueError:
            raise UEXCorpException(f"Request failed with status code {response.status_code}")
        
        return response
    
    async def api_call_cached(self, endpoint: str, method: str = "GET", data: dict | None = None):
        cache_path = self._file_cache_path / endpoint
        if cache_path.exists() and not await self._is_cache_stale(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        
        response = await self.api_call(endpoint, method, data)

        # ensure cache directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_path, "w") as f:
            try:
                json.dump(response, f)
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
                pass
        
        return response

    async def _is_cache_stale(self, cache_path: Path) -> bool:
        if not cache_path.exists():
            return True
        
        return (time.time() - cache_path.stat().st_mtime) > self._cache_expiry

    async def fetch_commodities(self) -> list[dict[str, Any]]:
        return await self.api_call_cached("commodities")