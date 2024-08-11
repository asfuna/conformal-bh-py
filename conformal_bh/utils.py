from bisect import bisect_left

import numpy as np


def benjamini_hochberg(p_values, q: float = 0.05):
    """
    Perform the Benjamini-Hochberg procedure
    https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Hochberg_procedure

    :param p_values: array of p-values from multiple hypothesis tests.
    :param q: The desired false discovery rate level.

    :return: A numpy array of booleans indicating which hypotheses are rejected.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    bh_thresholds = np.arange(1, n + 1) * q / n
    below_threshold = sorted_p_values <= bh_thresholds
    reject = np.zeros(n, dtype=bool)
    if np.any(below_threshold):
        max_below_threshold = np.where(below_threshold)[0].max()
        reject[sorted_indices[:max_below_threshold + 1]] = True

    return reject


def conformal_p_values(calibration_scores, test_scores):
    """
    Perform the conformal p-value test
    :param calibration_scores: scores used for calibration
    :param test_scores: scores to test

    :return: A numpy array of p-values
    """
    calibration_scores = np.sort(calibration_scores)
    unique_cal_scores, counts = np.unique(calibration_scores, return_counts=True)
    cumulative_counts = np.cumsum(np.concatenate([np.zeros(1), counts]))
    counts_as_dict = dict(zip(unique_cal_scores, counts))
    n_calibration = len(calibration_scores)
    n_test = len(test_scores)
    p_values = np.zeros(n_test)

    for j in range(n_test):
        pos = bisect_left(calibration_scores, test_scores[j])
        strictly_lower = cumulative_counts[pos]
        tie_breaker = np.random.uniform() * (counts_as_dict.get(test_scores[j], 0) + 1)
        p_values[j] = (strictly_lower + tie_breaker) / (n_calibration + 1)

    return p_values
