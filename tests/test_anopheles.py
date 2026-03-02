from types import SimpleNamespace

import numpy as np
import pandas as pd

from malariagen_data.anopheles import AnophelesDataResource
from malariagen_data.util import CacheMiss


class DummyAnophelesDataResource(AnophelesDataResource):
    @property
    def _xpehh_gwss_cache_name(self):
        return "dummy_xpehh_gwss_cache"

    @property
    def _ihs_gwss_cache_name(self):
        return "dummy_ihs_gwss_cache"

    @property
    def _roh_hmm_cache_name(self):
        return "dummy_roh_hmm_cache"

    def __init__(self):
        self._log = SimpleNamespace(debug=lambda *args, **kwargs: None)
        self._cache = {}
        self.block_jackknife_calls = 0

    def sample_metadata(self, sample_sets=None, sample_query=None):  # noqa: ARG002
        data = pd.DataFrame(
            {
                "sample_id": ["s1", "s2", "s3"],
                "cohort_admin1_year": ["cohort_a", "cohort_a", "cohort_b"],
                "taxon": ["gambiae", "gambiae", "coluzzii"],
                "year": [2020, 2020, 2021],
                "month": [1, 2, 3],
                "country": ["Ghana", "Ghana", "Benin"],
                "admin1_iso": ["GH-AA", "GH-AA", "BJ-AK"],
                "admin1_name": ["Accra", "Accra", "Atlantique"],
                "admin2_name": ["Accra", "Accra", "Abomey-Calavi"],
                "longitude": [-0.2, -0.4, 2.3],
                "latitude": [5.6, 5.8, 6.4],
            }
        )
        if sample_query is not None:
            return data.query(sample_query).reset_index(drop=True)
        return data

    def snp_allele_counts(self, **kwargs):  # noqa: ARG002
        return np.array([[2, 0], [1, 1], [0, 2]])

    def _block_jackknife_cohort_diversity_stats(
        self, *, cohort_label, ac, n_jack, confidence_level  # noqa: ARG002
    ):
        self.block_jackknife_calls += 1
        return {
            "cohort": cohort_label,
            "theta_pi": 0.123,
            "theta_pi_estimate": 0.124,
            "theta_pi_bias": 0.001,
            "theta_pi_std_err": 0.01,
            "theta_pi_ci_err": 0.02,
            "theta_pi_ci_low": 0.1,
            "theta_pi_ci_upp": 0.14,
            "theta_w": 0.111,
            "theta_w_estimate": 0.112,
            "theta_w_bias": 0.001,
            "theta_w_std_err": 0.01,
            "theta_w_ci_err": 0.02,
            "theta_w_ci_low": 0.09,
            "theta_w_ci_upp": 0.13,
            "tajima_d": 0.3,
            "tajima_d_estimate": 0.31,
            "tajima_d_bias": 0.01,
            "tajima_d_std_err": 0.05,
            "tajima_d_ci_err": 0.1,
            "tajima_d_ci_low": 0.2,
            "tajima_d_ci_upp": 0.4,
        }

    def results_cache_get(self, *, name, params):
        key = (name, repr(params))
        if key not in self._cache:
            raise CacheMiss
        return self._cache[key]

    def results_cache_set(self, *, name, params, results):
        key = (name, repr(params))
        self._cache[key] = results


def test_cohort_diversity_stats_uses_cache():
    api = DummyAnophelesDataResource()

    stats1 = api.cohort_diversity_stats(
        cohort="cohort_a",
        cohort_size=2,
        region="2L",
        n_jack=10,
        confidence_level=0.95,
    )
    stats2 = api.cohort_diversity_stats(
        cohort="cohort_a",
        cohort_size=2,
        region="2L",
        n_jack=10,
        confidence_level=0.95,
    )

    assert api.block_jackknife_calls == 1
    pd.testing.assert_series_equal(stats1.sort_index(), stats2.sort_index())

