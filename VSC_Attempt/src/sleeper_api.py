import requests
from typing import Any, Dict, Optional

BASE = 'https://api.sleeper.app/v1'


class SleeperAPIError(Exception):
    pass


class SleeperClient:
    def __init__(self, base_url: str = BASE, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=self.timeout)
        if not resp.ok:
            raise SleeperAPIError(f"GET {url} failed: {resp.status_code} {resp.text}")
        return resp.json()

    def get_players(self) -> Dict[str, Any]:
        """Return mapping of player_id -> player object"""
        return self._get('/players/nfl')

    def get_league(self, league_id: str) -> Dict[str, Any]:
        return self._get(f'/league/{league_id}')

    def get_rosters(self, league_id: str) -> Any:
        return self._get(f'/league/{league_id}/rosters')

    def get_matchups(self, league_id: str, week: Optional[int] = None) -> Any:
        # Sleeper API expects the week in the path: /league/{league_id}/matchups/{week}
        if week is not None:
            path = f'/league/{league_id}/matchups/{week}'
            return self._get(path)
        # Fallback to base path (may return latest or 404 depending on API behavior)
        path = f'/league/{league_id}/matchups'
        return self._get(path)

    def get_league_users(self, league_id: str) -> Any:
        """Return the list of users in a league, including display_name and user_id."""
        return self._get(f'/league/{league_id}/users')

    def get_week_stats(self, season: int, week: int) -> Any:
        """Return raw weekly player stats for given season/week.

        Sleeper endpoint structure (subject to change): /stats/nfl/regular/{season}/{week}
        We add a timeout and simple error propagation; caller will normalize keys.
        """
        path = f'/stats/nfl/regular/{season}/{week}'
        params = {'season_type': 'regular'}
        return self._get(path, params=params)

    def get_user(self, user_id: str) -> Any:
        return self._get(f'/user/{user_id}')

    def get_leagues_for_user(self, user_id: str, season: str) -> Any:
        return self._get(f'/users/{user_id}/leagues/nfl/{season}')