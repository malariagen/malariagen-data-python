import numpy as np
import pandas as pd
import pytest

from malariagen_data import Ag3


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture, tmp_path):
    data_path = ag3_sim_fixture.bucket_path.as_posix()
    return Ag3(
        url=data_path,
        public_url=data_path,
        pre=True,
        check_location=False,
        bokeh_output_notebook=False,
        results_cache=tmp_path.as_posix(),
    )


def test_cohort_diversity_stats_uses_cache(ag3_sim_api, monkeypatch):
    api = ag3_sim_api
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = str(np.random.choice(all_sample_sets))
    df_samples = api.sample_metadata(sample_sets=[sample_set])
    cohort_sample_ids = df_samples["sample_id"].head(10).to_list()
    cohort_size = min(5, len(cohort_sample_ids))
    if cohort_size < 2:
        pytest.skip("not enough samples in simulated cohort")

    params = dict(
        cohort=("cache_test", f"sample_id in {cohort_sample_ids!r}"),
        cohort_size=cohort_size,
        region=str(np.random.choice(api.contigs)),
        sample_sets=[sample_set],
        random_seed=42,
        n_jack=10,
        confidence_level=0.95,
    )

    stats_first = api.cohort_diversity_stats(**params)

    def _unexpected_recompute(*args, **kwargs):  # noqa: ARG001, ARG002
        raise AssertionError(
            "cohort_diversity_stats recomputed instead of loading from cache"
        )

    monkeypatch.setattr(
        api, "_block_jackknife_cohort_diversity_stats", _unexpected_recompute
    )
    stats_second = api.cohort_diversity_stats(**params)

    pd.testing.assert_series_equal(
        stats_first.sort_index(),
        stats_second.sort_index(),
        check_dtype=False,
    )
