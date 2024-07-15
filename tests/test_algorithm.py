import unittest

import numpy as np

from conformal_bh import nonconformity_scores
from conformal_bh.algorithm import evaluate


class TestAlgorithm(unittest.TestCase):
    def test_evaluate(self):
        np.random.seed(834230)

        x = np.random.uniform(low=-1, high=1, size=(100, 5))
        mu_x = x[:, 0] * x[:, 1] + x[:, 2] / np.exp(x[:, 3])
        y = mu_x + np.random.normal(scale=0.25, size=100)

        def model(z):
            return z[0] * z[1] + z[2] / np.exp(z[3])

        score_function = nonconformity_scores.CliffModelNonconformityScore(
            model=model,
            cliff=0,
            max_model=4
        )

        np.testing.assert_array_equal(
            evaluate(zip(x[:90], y[:90]), zip(x[90:], np.zeros(10)), score_function),
            np.array([model(z) > 0.15 for z in x[90:]])
        )


if __name__ == "__main__":
    unittest.main()
