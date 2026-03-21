import secrets
import pytest

GLOBAL_SEED = secrets.randbits(128)


# The report header needs to be in the top-level
# conftest.py in order to be printed in all cases.
def pytest_report_header(config):
    return f"global seed: {GLOBAL_SEED}"


@pytest.fixture(scope="session", name="global_seed")
def global_seed_fixture():
    return GLOBAL_SEED
