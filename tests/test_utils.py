import unittest
from unittest.mock import patch

import numpy as np

from conformal_bh import utils


class TestUtils(unittest.TestCase):
    def test_benjamini_hochberg(self):
        p_values = np.array([0.01, 0.04, 0.03, 0.002, 0.005, 0.03, 0.12, 0.06])
        q = 0.05

        rejected_hypotheses = utils.benjamini_hochberg(p_values, q)
        np.testing.assert_array_equal(rejected_hypotheses, np.array([True, False, True, True, True, True, False, False]))

    def test_conformal_p_values(self):
        test_scores = np.array([1.5, 2.5, 3.5])
        calibration_scores = np.array([1.0, 2.0, 2.5, 3.0])

        # Patch np.random.uniform to return a consistent value for testing purposes
        with patch('numpy.random.uniform', return_value=np.array([0.5])):
            expected_p_values = np.array([
                (np.sum(calibration_scores < 1.5) + 0.5 * (np.sum(calibration_scores == 1.5) + 1)) / (
                            len(calibration_scores) + 1),
                (np.sum(calibration_scores < 2.5) + 0.5 * (np.sum(calibration_scores == 2.5) + 1)) / (
                            len(calibration_scores) + 1),
                (np.sum(calibration_scores < 3.5) + 0.5 * (np.sum(calibration_scores == 3.5) + 1)) / (
                            len(calibration_scores) + 1)
            ])
            result = utils.conformal_p_values(calibration_scores, test_scores)
            np.testing.assert_almost_equal(result, expected_p_values, decimal=6)


if __name__ == "__main__":
    unittest.main()
