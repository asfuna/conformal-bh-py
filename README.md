# conformal-bh-py

## Description
`conformal-bh-py` is a Python package designed for evaluating multiple hypotheses using a pre-trained model, and a set of calibration data.
It uses conformal prediction to calculate p-values for the hypotheses, and applies the Benjamini-Hochberg algorithm to control the False Discovery Rate (FDR).

## Acknowledgment
This package is based on the work by [Ying Jin and Emmanuel J. Candès (2022)](https://arxiv.org/abs/2210.01408), using code from their [simulation project](https://github.com/ying531/selcf_paper).

## Installation
To install `conformal-bh-py`, you can use pip:
```bash
pip install conformal-bh-py
```

## Usage
Here is a basic example of how to use the `evaluate` function provided by this package:

### Example
```python
from conformal_bh import evaluate, ModelNonconformityScore

# Example calibration data
calibration_data = [(1, 3), (2, 5), (3, 7), (4, 9)]

# Example hypotheses
hypothesises = [(1, 2), (2, 4), (3, 6), (4, 8)]

# Initialize your score function
nonconformity_score = ModelNonconformityScore(model=...)

# Evaluate the hypotheses
rejected = evaluate(calibration_data, hypothesises, nonconformity_score, q=0.05)

print("Rejected hypotheses:", rejected)
```

### Detailed Description of the `evaluate` Function
The `evaluate` function performs a conformal prediction algorithm to evaluate a set of hypotheses using calibration data and a provided score function.

#### Parameters:
- `calibration_data`: A list of (x_i, y_i) tuples representing the calibration dataset.  
  - `x_i`: Independent variable(s) (features).  
  - `y_i`: Dependent variable (target).
- `hypothesises`: A list of (x_i, c_i) tuples representing the hypotheses to be evaluated.  
  - `x_i`: Independent variable(s) (features).  
  - `c_i`: Hypothesized threshold for the dependent variable.
- `nonconformity_score`: NonconformityScore
  - An instance of a class that has a `get_score` method.
  - The `get_score` method should take two arguments (x, y) and return a real-valued score.
  - A score V should satisfy V(x, y) <= V(x, z) where y <= z
- `q`: The target False Discovery Rate (FDR) level for the Benjamini-Hochberg procedure (default is 0.05).

#### Returns:
- A list of booleans indicating which hypotheses are rejected.  
  - `True`: The hypothesis is rejected.  
  - `False`: The hypothesis is not rejected.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
