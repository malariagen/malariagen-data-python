import numpy as np
from malariagen_data.util import compute_missing_rate, compute_informative_sites


def test_compute_missing_rate():
    gt = np.array([[0, 1], [-1, -1]])
    assert compute_missing_rate(gt) == 0.5


def test_compute_informative_sites():
    gt = np.array([[0, 1], [-1, -1]])
    assert compute_informative_sites(gt) == 1