import unittest

import numpy as np
import pandas as pd

from fintda.ftda import FinTDA


class FinanceTDATest(unittest.TestCase):
    def setUp(self):
        # Create sample returns data
        dates = pd.date_range(start='2021-01-01', end='2022-01-10')
        returns = pd.DataFrame(np.random.randn(len(dates), 3), index=dates, columns=['Asset1', 'Asset2', 'Asset3'])
        self.n = len(returns)

        # Set sample weights
        weights = [0.3, 0.4, 0.3]

        # Create FinTDA instance
        self.ftda = FinTDA(returns, weights)

    def test_compute_dgm(self):
        # Test compute_dgm method
        dgm = self.ftda.compute_dgm()
        self.assertIsNotNone(dgm)
        self.assertIsInstance(dgm, list)
        self.assertEqual(len(dgm), 3)

    def test_compute_moving_dgm(self):
        # Test compute_moving_dgm method
        moving_dgm = self.ftda.compute_moving_dgm()
        self.assertIsNotNone(moving_dgm)
        self.assertIsInstance(moving_dgm, pd.Series)
        self.assertEqual(len(moving_dgm), self.n - 39)

    def test_attributes(self):
        # Test attributes
        self.assertIsNotNone(self.ftda.returns)
        self.assertIsNotNone(self.ftda.weights)
        self.assertIsNotNone(self.ftda._raw_data)
        self.assertIsNotNone(self.ftda.n)
        self.assertIsNotNone(self.ftda._portfolio_volatility)
        self.assertIsNotNone(self.ftda._mean_pnl)
        self.assertIsNotNone(self.ftda._volatility)
        self.assertIsNotNone(self.ftda.info)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
