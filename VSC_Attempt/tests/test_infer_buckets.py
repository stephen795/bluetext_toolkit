import unittest
from src.sleeper_mapper import infer_buckets_from_rec_and_yards


class TestInferBuckets(unittest.TestCase):
    def test_infer_buckets_basic(self):
        rec = 10
        rec_yds = 120.0
        buckets = infer_buckets_from_rec_and_yards(rec, rec_yds)
        # counts sum to rec
        self.assertEqual(sum(buckets.values()), rec)
        # estimated yards from buckets roughly equals rec_yds
        means = {
            'rec_0_4': 2.0,
            'rec_5_9': 7.0,
            'rec_10_19': 14.5,
            'rec_20_29': 24.5,
            'rec_30_39': 34.5,
            'rec_40p': 50.0,
        }
        est = sum(buckets[k] * means[k] for k in buckets)
        # allow some deviation (±20%) since it's heuristic
        self.assertTrue(abs(est - rec_yds) / rec_yds < 0.2)


if __name__ == '__main__':
    unittest.main()
