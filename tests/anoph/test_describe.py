import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.describe import AnophelesDescribe


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesDescribe(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesDescribe(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_describe_api_returns_dataframe(fixture, api):
    """Test that describe_api returns a DataFrame with expected columns."""
    df = api.describe_api()
    assert isinstance(df, pd.DataFrame)
    assert "method" in df.columns
    assert "summary" in df.columns
    assert "category" in df.columns
    assert len(df) > 0


@parametrize_with_cases("fixture,api", cases=".")
def test_describe_api_no_private_methods(fixture, api):
    """Test that describe_api does not include private or dunder methods."""
    df = api.describe_api()
    for method_name in df["method"]:
        assert not method_name.startswith(
            "_"
        ), f"Private method {method_name!r} should not appear in describe_api output"


@parametrize_with_cases("fixture,api", cases=".")
def test_describe_api_category_filter(fixture, api):
    """Test filtering by category."""
    for category in ("data", "analysis", "plot"):
        df = api.describe_api(category=category)
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert all(df["category"] == category)


@parametrize_with_cases("fixture,api", cases=".")
def test_describe_api_invalid_category(fixture, api):
    """Test that an invalid category raises ValueError."""
    with pytest.raises(ValueError, match="Invalid category"):
        api.describe_api(category="invalid")


@parametrize_with_cases("fixture,api", cases=".")
def test_describe_api_known_methods(fixture, api):
    """Test that some known methods appear in the output."""
    df = api.describe_api()
    method_names = set(df["method"])
    # These methods should exist on AnophelesDescribe (inherited from AnophelesBase).
    assert "describe_api" in method_names
    assert "sample_sets" in method_names


@parametrize_with_cases("fixture,api", cases=".")
def test_describe_api_summaries_not_empty(fixture, api):
    """Test that at least some methods have non-empty summaries."""
    df = api.describe_api()
    non_empty = df[df["summary"] != ""]
    assert len(non_empty) > 0, "Expected at least some methods to have summaries"


def test_categorize_method():
    """Test the static _categorize_method helper."""
    assert AnophelesDescribe._categorize_method("plot_pca") == "plot"
    assert AnophelesDescribe._categorize_method("plot_heterozygosity") == "plot"
    assert AnophelesDescribe._categorize_method("sample_metadata") == "data"
    assert AnophelesDescribe._categorize_method("snp_calls") == "data"
    assert AnophelesDescribe._categorize_method("genome_sequence") == "data"
    assert AnophelesDescribe._categorize_method("lookup_release") == "data"
    assert AnophelesDescribe._categorize_method("diversity_stats") == "analysis"
    assert AnophelesDescribe._categorize_method("cohort_diversity_stats") == "analysis"


def test_extract_summary():
    """Test the static _extract_summary helper."""

    def dummy_func():
        """This is a test summary.

        More details here.
        """
        pass

    summary = AnophelesDescribe._extract_summary(dummy_func)
    assert summary == "This is a test summary."

    def no_doc_func():
        pass

    summary = AnophelesDescribe._extract_summary(no_doc_func)
    assert summary == ""
