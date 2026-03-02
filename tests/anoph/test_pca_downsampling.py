from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from malariagen_data.anoph.pca import AnophelesPca
from malariagen_data.util import CacheMiss


class DummyAnophelesPca(AnophelesPca):
    def __init__(self):
        self._log = SimpleNamespace(debug=lambda *args, **kwargs: None)
        self._captured_pca_params = None

    def _prep_sample_selection_cache_params(
        self, *, sample_sets, sample_query, sample_query_options, sample_indices  # noqa: ARG002
    ):
        return ["dummy_set"], None

    def _prep_region_cache_param(self, *, region):
        return region

    def _prep_optional_site_mask_param(self, *, site_mask):
        return site_mask

    def results_cache_get(self, *, name, params):  # noqa: ARG002
        raise CacheMiss

    def results_cache_set(self, *, name, params, results):  # noqa: ARG002
        return None

    def sample_metadata(
        self,
        sample_sets=None,  # noqa: ARG002
        sample_query=None,
        sample_query_options=None,  # noqa: ARG002
        sample_indices=None,
    ):
        df = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(6)],
                "country": ["Ghana", "Ghana", "Ghana", "Benin", "Benin", "Benin"],
                "location": ["x", "x", "x", "y", "y", "y"],
            }
        )
        if sample_query is not None:
            df = df.query(sample_query).reset_index(drop=True)
        if sample_indices is not None:
            df = df.iloc[sample_indices].reset_index(drop=True)
        return df

    def _pca(self, **params):
        self._captured_pca_params = params
        sample_indices = params["sample_indices"]
        all_samples = np.asarray([f"s{i}" for i in range(6)], dtype="U")
        if sample_indices is None:
            samples = all_samples
        else:
            samples = all_samples[np.asarray(sample_indices, dtype=int)]
        n_components = params["n_components"]
        n_samples = samples.shape[0]
        return {
            "samples": samples,
            "coords": np.zeros((n_samples, n_components)),
            "evr": np.ones(n_components),
            "loc_keep_fit": np.ones(n_samples, dtype=bool),
        }


def test_pca_cohort_size_query_requires_cohort_size():
    api = DummyAnophelesPca()

    with pytest.raises(ValueError, match="cohort_size must be provided"):
        api.pca(
            region="2L",
            n_snps=4,
            n_components=2,
            cohort_size_query="country",
        )


def test_pca_cohort_size_query_downsamples_per_cohort():
    api = DummyAnophelesPca()

    pca_df, _ = api.pca(
        region="2L",
        n_snps=4,
        n_components=2,
        cohort_size=2,
        cohort_size_query="country",
        random_seed=42,
    )

    captured = api._captured_pca_params
    assert captured is not None
    assert captured["cohort_size_query"] == "country"
    assert captured["sample_indices"] is not None

    selected_df = api.sample_metadata().iloc[captured["sample_indices"]]
    selected_counts = selected_df["country"].value_counts()
    assert selected_counts["Ghana"] == 2
    assert selected_counts["Benin"] == 2
    assert len(pca_df) == 4
