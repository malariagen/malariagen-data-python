from .rng_util import GLOBAL_SEED


# The report header needs to be in the top-level
# conftest.py in order to be shown in all cases.
def pytest_report_header(config):
    return f"global seed: {GLOBAL_SEED}"
