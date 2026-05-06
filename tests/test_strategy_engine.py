import unittest
from datetime import datetime
from alpha_factory.strategy_engine import StrategyEngine
import os

class TestStrategyEngine(unittest.TestCase):
    def setUp(self):
        from research_lab.data_engine import DataEngine
        self.data_engine = DataEngine(":memory:")
        self.engine = StrategyEngine(data_provider=self.data_engine)
        
        # Warm up engine with synthetic data
        self.data_engine.generate_synthetic_pit_data(
            self.engine.config['universe']['tickers']
        )

    def test_get_current_rankings(self):
        # Use a historical date to ensure data is available
        as_of = datetime(2024, 1, 10)
        rankings = self.engine.get_current_rankings(as_of)
        
        self.assertIn("status", rankings)
        self.assertIn("ladder", rankings)
        self.assertIn("belief_score", rankings)
        self.assertIn("signal_energy", rankings)
        
        if rankings["status"] == "OK":
            self.assertTrue(len(rankings["ladder"]) > 0)
            self.assertTrue(rankings["belief_score"] > 0)
            self.assertTrue(rankings["signal_energy"] >= 0)
            print(f"✅ Test rankings: {len(rankings['ladder'])} tickers found.")
        else:
            print(f"⚠️ Test rankings status: {rankings['status']}")

if __name__ == "__main__":
    unittest.main()
