"""
Client for API-Football (v3) with a local cache.

Every response is saved as JSON in data/raw/ so a given request is only paid once.
The API key is read from the API_FOOTBALL_KEY environment variable, or from .env.
"""

import json
import os
import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

BASE_URL = "https://v3.football.api-sports.io"
PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_DIR / "data" / "raw"

# The free plan allows about 10 requests per minute: space calls out and retry on 429.
MIN_INTERVAL_SECONDS = 6.5
MAX_RETRIES = 5


def load_api_key() -> str:
    """Return the API key from the environment or from the .env file."""
    key = os.getenv("API_FOOTBALL_KEY")
    if key:
        return key
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                name, value = line.split("=", 1)
                if name.strip() == "API_FOOTBALL_KEY":
                    return value.strip().strip("\"'")
    raise RuntimeError("API_FOOTBALL_KEY not found in environment or .env")


class ApiFootball:
    def __init__(self, api_key: Optional[str] = None, cache_dir: Path = CACHE_DIR):
        self.api_key = api_key or load_api_key()
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0  # real HTTP calls in this session (cache hits do not count)
        self._last_call = 0.0

    def _cache_path(self, endpoint: str, params: Dict) -> Path:
        query = urllib.parse.urlencode(sorted(params.items()))
        digest = hashlib.sha1(f"{endpoint}?{query}".encode()).hexdigest()[:12]
        safe_name = endpoint.strip("/").replace("/", "_")
        return self.cache_dir / f"{safe_name}_{digest}.json"

    def _fetch(self, url: str) -> Dict:
        """One throttled HTTP GET, retrying on 429 with a growing pause."""
        for attempt in range(MAX_RETRIES):
            wait = MIN_INTERVAL_SECONDS - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            request = urllib.request.Request(url, headers={"x-apisports-key": self.api_key})
            self._last_call = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.calls_made += 1
                    return json.load(response)
            except urllib.error.HTTPError as e:
                if e.code != 429 or attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(60)  # per-minute window: wait for it to reset
        raise RuntimeError("unreachable")

    def cached(self, endpoint: str, **params) -> Optional[Dict]:
        """Return the cached body for a request, or None. Never calls the API."""
        path = self._cache_path(endpoint, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def get(self, endpoint: str, **params) -> Dict:
        """GET an endpoint, returning the full JSON body (cached on disk)."""
        path = self._cache_path(endpoint, params)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        query = urllib.parse.urlencode(params)
        body = self._fetch(f"{BASE_URL}/{endpoint.strip('/')}?{query}")

        if body.get("errors"):
            raise RuntimeError(f"API error on {endpoint}: {body['errors']}")

        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        return body

    def status(self) -> Dict:
        """Account status and today's request quota (not cached: always fresh)."""
        return self._fetch(f"{BASE_URL}/status")["response"]
