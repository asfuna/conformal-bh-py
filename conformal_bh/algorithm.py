from conformal_bh.nonconformity_scores import NonconformityScore
from conformal_bh.utils import conformal_p_values, benjamini_hochberg


def evaluate(calibration_data, hypotheses, nonconformity_score: NonconformityScore, q=0.05):
    """
    Evaluate multiple hypotheses using a score function and calibration data.

    This function performs a conformal prediction algorithm to evaluate a set of hypotheses.
    It uses calibration data and a provided score function to calculate p-values for each hypothesis
    and applies the Benjamini-Hochberg procedure to control the False Discovery Rate (FDR).

    Parameters:
    -----------
    calibration_data : list of tuples
        A list of (x_i, y_i) tuples representing the calibration dataset.
        - x_i : Independent variable(s) (features).
        - y_i : Dependent variable (target).

    hypotheses : list of tuples
        A list of (x_i, c_i) tuples representing the hypotheses to be evaluated.
        - x_i : Independent variable(s) (features).
        - c_i : Threshold for the dependent variable.

    nonconformity_score : NonconformityScore
        An instance of a class that has a `get_score` method.
        The `get_score` method should take two arguments (x, y) and return a real-valued score.
        - A score V should satisfy V(x, y) <= V(x, z) where y <= z

    q : float, optional
        The target False Discovery Rate (FDR) level for the Benjamini-Hochberg procedure (default is 0.05).

    Returns:
    --------
    list of bool
        A list of booleans indicating which hypotheses are rejected.
        - True : The hypothesis is rejected.
        - False : The hypothesis is not rejected.

    Example:
    --------
    >>> from conformal_bh.nonconformity_scores import ModelNonconformityScore
    >>> calibration_data = [(x1, y1), (x2, y2), ...]
    >>> hypotheses = [(x1, c1), (x2, c2), ...]
    >>> nonconformity_score = ModelNonconformityScore(model=...)
    >>> rejected = evaluate(calibration_data, hypotheses, nonconformity_score, q=0.05)
    """
    calibration_scores = [nonconformity_score.get_score(x, y) for x, y in calibration_data]
    test_scores = [nonconformity_score.get_score(x, c) for x, c in hypotheses]
    p_values = conformal_p_values(calibration_scores, test_scores)
    rejected = benjamini_hochberg(p_values, q=q)
    return rejected
