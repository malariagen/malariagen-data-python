import numpy as np
import pytest
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


def test_haplotype_frequencies():
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
    from malariagen_data.anopheles import _haplotype_frequencies

    f = _haplotype_frequencies(h1)
    assert isinstance(f, dict)
    vals = np.array(list(f.values()))
    vals.sort()
    assert np.all(vals >= 0)
    assert np.all(vals <= 1)
    assert_allclose(vals, np.array([0.2, 0.2, 0.2, 0.4]))


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
    from malariagen_data.anopheles import _haplotype_joint_frequencies

    f = _haplotype_joint_frequencies(h1, h2)
    assert isinstance(f, dict)
    vals = np.array(list(f.values()))
    vals.sort()
    assert np.all(vals >= 0)
    assert np.all(vals <= 1)
    assert_allclose(vals, np.array([0, 0, 0, 0, 0.04, 0.16]))


@pytest.mark.parametrize(
    "subclass, sample_query, contig, analysis, sample_sets",
    [
        (Ag3, "country == 'Ghana'", "3L", "gamb_colu", "3.0"),
        (Af1, "country == 'Ghana'", "X", "funestus", "1.0"),
    ],
)
def test_h12_calibration(subclass, sample_query, contig, analysis, sample_sets):
    anoph = setup_subclass_cached(subclass)

    window_sizes = (10_000, 20_000)
    calibration_runs = anoph.h12_calibration(
        contig=contig,
        analysis=analysis,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_sizes=window_sizes,
        cohort_size=20,
    )

    # check dataset
    assert isinstance(calibration_runs, dict)
    assert isinstance(calibration_runs[str(window_sizes[0])], np.ndarray)

    # check dimensions
    assert len(calibration_runs) == len(window_sizes)

    # check keys
    assert list(calibration_runs.keys()) == [str(win) for win in window_sizes]


@pytest.mark.parametrize(
    "subclass, sample_query, contig, site_mask, sample_sets",
    [
        (Ag3, "country == 'Ghana'", "3L", "gamb_colu", "3.0"),
        (Af1, "country == 'Ghana'", "X", "funestus", "1.0"),
    ],
)
def test_g123_calibration(subclass, sample_query, contig, site_mask, sample_sets):
    anoph = setup_subclass_cached(subclass)

    window_sizes = (10_000, 20_000)
    calibration_runs = anoph.g123_calibration(
        contig=contig,
        sites=site_mask,
        site_mask=site_mask,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_sizes=window_sizes,
        min_cohort_size=20,
        max_cohort_size=30,
    )

    # check dataset
    assert isinstance(calibration_runs, dict)
    assert isinstance(calibration_runs[str(window_sizes[0])], np.ndarray)

    # check dimensions
    assert len(calibration_runs) == len(window_sizes)

    # check keys
    assert list(calibration_runs.keys()) == [str(win) for win in window_sizes]
