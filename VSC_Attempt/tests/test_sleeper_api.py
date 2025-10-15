import unittest
import responses
from src.sleeper_api import SleeperClient, BASE, SleeperAPIError

class TestSleeperClient(unittest.TestCase):
    @responses.activate
    def test_get_players(self):
        url = BASE + '/players/nfl'
        responses.add(responses.GET, url, json={'p1': {'full_name': 'Player One'}}, status=200)
        client = SleeperClient()
        res = client.get_players()
        self.assertIn('p1', res)

    @responses.activate
    def test_get_players_404(self):
        url = BASE + '/players/nfl'
        responses.add(responses.GET, url, json={'error': 'not found'}, status=404)
        client = SleeperClient()
        with self.assertRaises(SleeperAPIError):
            client.get_players()

if __name__ == '__main__':
    unittest.main()
