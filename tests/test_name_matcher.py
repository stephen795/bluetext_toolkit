import unittest
from src.name_matcher import find_best_match, normalize_name


class TestNameMatcher(unittest.TestCase):
    def test_exact_match(self):
        candidates = ['Tom Brady', 'Aaron Rodgers']
        self.assertEqual(find_best_match('Tom Brady', candidates), 'Tom Brady')

    def test_fuzzy_match(self):
        candidates = ['Tom Brady', 'Aaron Rodgers']
        self.assertEqual(find_best_match('T Brady', candidates), 'Tom Brady')

    def test_overrides(self):
        candidates = ['T.Brady', 'A.Rodgers']
        overrides = {normalize_name('tom brady'): 'T.Brady'}
        self.assertEqual(find_best_match('Tom Brady', candidates, overrides=overrides), 'T.Brady')


if __name__ == '__main__':
    unittest.main()
