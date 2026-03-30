import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def seed_random():
    np.random.seed(42)
