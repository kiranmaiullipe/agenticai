import unittest
from agent import detect_intent, answer
from rag import retrieve_policy

class AgentTests(unittest.TestCase):
    def test_routing(self):
        self.assertEqual(detect_intent("forecast demand"), "forecast")
        self.assertEqual(detect_intent("which products may stockout?"), "stockout_risks")
        self.assertEqual(detect_intent("how much should we reorder?"), "inventory_plan")
        self.assertEqual(detect_intent("promotion ideas"), "promo_recommendations")
        self.assertEqual(detect_intent("what does policy say?"), "policy")
    def test_policy_retrieval(self):
        hits = retrieve_policy("stockout risk", k=2)
        self.assertTrue(hits)
        self.assertIn("source", hits[0])
    def test_empty_question(self):
        self.assertEqual(answer(" ")["intent"], "unknown")

if __name__ == "__main__":
    unittest.main()
