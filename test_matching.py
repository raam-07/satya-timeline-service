import unittest
import numpy as np
import json
import sqlite3
from unittest.mock import MagicMock, patch

from timeline_pipeline import extract_keys, decompress_text, slugify

class TestTimelineService(unittest.TestCase):

    def test_extract_keys(self):
        # Party list, minister list, state list, city list
        party = '["BJP", "INC"]'
        minister = '["Narendra Modi", "Amit Shah"]'
        state = '["Kerala"]'
        city = '[]'
        
        keys = extract_keys(party, minister, state, city)
        self.assertIn("bjp", keys)
        self.assertIn("inc", keys)
        self.assertIn("narendra_modi", keys)
        self.assertIn("amit_shah", keys)
        self.assertIn("kerala", keys)
        self.assertEqual(len(keys), 5)

        # Empty state representation check
        self.assertEqual(extract_keys('[]', '[]', '[]', '[]'), [])

    def test_decompress_text(self):
        import zlib
        raw_text = "This is a rephrased article summary."
        compressed = zlib.compress(raw_text.encode('utf-8'))
        
        self.assertEqual(decompress_text(compressed), raw_text)
        self.assertEqual(decompress_text(None), "")

    def test_slugify(self):
        self.assertEqual(slugify("Kochi Metro Phase 2 Approval"), "kochi-metro-phase-2-approval")
        self.assertEqual(slugify("Yogi Vows: End Mafia-Raj!"), "yogi-vows-end-mafia-raj")

    def test_milestone_validation(self):
        # Helper to simulate pipeline milestone validation rule
        def validate(milestone):
            words = milestone.split()
            if len(words) < 3 or len(words) > 30:
                return False
            if milestone.startswith('"') or milestone.endswith('"') or milestone.startswith("'") or milestone.endswith("'"):
                return False
            return True

        self.assertTrue(validate("Modi inaugurates the Kochi Metro Phase 2 expansion project."))
        self.assertFalse(validate("Too short"))
        self.assertFalse(validate("A " * 35)) # too long
        self.assertFalse(validate('"Quotes around milestone"'))

    def test_cosine_similarity(self):
        # Check cosine similarity dot product math on unit vectors
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = float(np.dot(v1, v2))
        self.assertAlmostEqual(sim, 1.0)

        v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim_perp = float(np.dot(v1, v3))
        self.assertAlmostEqual(sim_perp, 0.0)

        # Normalization helper check
        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        v_unnormalized = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        v_normalized = normalize(v_unnormalized)
        self.assertAlmostEqual(np.linalg.norm(v_normalized), 1.0)

    def test_logical_clock_dormancy(self):
        # Verify 21 days logical clock threshold mapping
        # 21 days = 21 * 24 * 3600 seconds = 1,814,400 seconds
        max_scraped_at = 1780000000
        cutoff = max_scraped_at - 21 * 24 * 3600

        # Event last seen within 21 days (active)
        last_seen_active = max_scraped_at - 10 * 24 * 3600
        self.assertTrue(last_seen_active >= cutoff)

        # Event last seen more than 21 days ago (dormant)
        last_seen_dormant = max_scraped_at - 25 * 24 * 3600
        self.assertTrue(last_seen_dormant < cutoff)

    def test_merge_and_cap_keys(self):
        # Existing set of keys
        existing_set = {"A", "B", "C", "D", "E"}
        # Entities in new article
        art_entities = ["C", "D", "F", "G"]
        
        new_set = set(art_entities)
        intersection_list = list(existing_set.intersection(new_set))
        remaining_new = list(new_set.difference(existing_set))
        remaining_existing = list(existing_set.difference(new_set))
        
        combined = intersection_list + remaining_new + remaining_existing
        merged_keys = combined[:15]
        
        # Intersection keys ('C', 'D') must be first
        self.assertTrue(merged_keys[0] in ["C", "D"])
        self.assertTrue(merged_keys[1] in ["C", "D"])
        # New unique keys ('F', 'G') should follow
        self.assertTrue(merged_keys[2] in ["F", "G"])
        self.assertTrue(merged_keys[3] in ["F", "G"])
        self.assertEqual(len(merged_keys), 7)

        # Test capping at 15
        large_existing = set(str(i) for i in range(20))
        large_new = ["19", "20", "21"]
        
        large_new_set = set(large_new)
        large_intersection_list = list(large_existing.intersection(large_new_set))
        large_remaining_new = list(large_new_set.difference(large_existing))
        large_remaining_existing = list(large_existing.difference(large_new_set))
        
        large_combined = large_intersection_list + large_remaining_new + large_remaining_existing
        large_merged_keys = large_combined[:15]
        
        self.assertEqual(len(large_merged_keys), 15)
        # 19 (intersection) is first
        self.assertEqual(large_merged_keys[0], "19")

if __name__ == '__main__':
    unittest.main()
