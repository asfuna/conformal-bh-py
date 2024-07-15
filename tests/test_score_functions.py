import unittest

from conformal_bh import nonconformity_scores


class TestScoreFunctions(unittest.TestCase):
    def test_cliff_model_no_max(self):
        score_function = nonconformity_scores.CliffModelNonconformityScore(
            model=lambda x: x,
            cliff=0
        )
        self.assertEqual(score_function.get_score(1, 2), 1)
        self.assertEqual(score_function.get_score(1, -2), -1)

    def test_cliff_model_with_max(self):
        score_function = nonconformity_scores.CliffModelNonconformityScore(
            model=lambda x: x,
            cliff=0,
            max_model=5
        )
        self.assertEqual(score_function.get_score(1, 2), 9)
        self.assertEqual(score_function.get_score(1, -2), -1)


if __name__ == "__main__":
    unittest.main()
