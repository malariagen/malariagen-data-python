import numpy as np
from numpy.testing import assert_allclose

from malariagen_data import Af1, Ag3
from malariagen_data.af1 import GCS_URL as AF1_GCS_URL
from malariagen_data.ag3 import GCS_URL as AG3_GCS_URL

expected_cohort_cols = (
    "country_iso",
    "admin1_name",
    "admin1_iso",
    "admin2_name",
    "taxon",
    "cohort_admin1_year",
    "cohort_admin1_month",
    "cohort_admin2_year",
    "cohort_admin2_month",
)


def setup_subclass(subclass, url=None, **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        return subclass(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return subclass(url=url, **kwargs)


def setup_subclass_cached(subclass, **kwargs):
    if subclass == Ag3:
        url = f"simplecache::{AG3_GCS_URL}"
    elif subclass == Af1:
        url = f"simplecache::{AF1_GCS_URL}"
    else:
        raise ValueError
    return setup_subclass(subclass, url=url, **kwargs)


def test_haplotype_joint_frequencies():
    h1 = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype="i1",
    )
    h2 = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype="i1",
    )
    from malariagen_data.anoph.h1x import haplotype_joint_frequencies

    f = haplotype_joint_frequencies(h1, h2)
    assert isinstance(f, dict)
    vals = np.array(list(f.values()))
    vals.sort()
    assert np.all(vals >= 0)
    assert np.all(vals <= 1)
    assert_allclose(vals, np.array([0, 0, 0, 0, 0.04, 0.16]))
