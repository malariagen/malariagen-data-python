import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

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


@pytest.mark.parametrize(
    "subclass, sample_sets, universal_fields, transcript, site_mask, cohorts_analysis, expected_snp_count",
    [
        (
            Ag3,
            "3.0",
            [
                "pass_gamb_colu_arab",
                "pass_gamb_colu",
                "pass_arab",
                "label",
            ],
            "AGAP004707-RD",
            "gamb_colu",
            "20211101",
            16526,
        ),
        (
            Af1,
            "1.0",
            [
                "pass_funestus",
                "label",
            ],
            "LOC125767311_t2",
            "funestus",
            "20221129",
            4221,
        ),
    ],
)
def test_snp_allele_frequencies__str_cohorts(
    subclass,
    sample_sets,
    universal_fields,
    transcript,
    site_mask,
    cohorts_analysis,
    expected_snp_count,
):
    anoph = setup_subclass_cached(subclass, cohorts_analysis=cohorts_analysis)

    cohorts = "admin1_month"
    min_cohort_size = 10
    df = anoph.snp_allele_frequencies(
        transcript=transcript,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
        effects=False,
    )
    df_coh = anoph.cohorts_metadata(sample_sets=sample_sets)
    coh_nm = "cohort_" + cohorts
    coh_counts = df_coh[coh_nm].dropna().value_counts()
    cohort_labels = coh_counts[coh_counts >= min_cohort_size].index.to_list()
    frq_cohort_labels = ["frq_" + s for s in cohort_labels]
    expected_fields = universal_fields + frq_cohort_labels + ["max_af"]

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == ["contig", "position", "ref_allele", "alt_allele"]
    assert len(df) == expected_snp_count


@pytest.mark.parametrize(
    "subclass, transcript, sample_set",
    [
        (
            Ag3,
            "AGAP004707-RD",
            "AG1000G-FR",
        ),
        (
            Af1,
            "LOC125767311_t2",
            "1229-VO-GH-DADZIE-VMF00095",
        ),
    ],
)
def test_snp_allele_frequencies__dup_samples(
    subclass,
    transcript,
    sample_set,
):
    # Expect automatically deduplicate any sample sets.
    anoph = setup_subclass_cached(subclass)
    df = anoph.snp_allele_frequencies(
        transcript=transcript,
        cohorts="admin1_year",
        sample_sets=[sample_set],
    )
    df_dup = anoph.snp_allele_frequencies(
        transcript=transcript,
        cohorts="admin1_year",
        sample_sets=[sample_set, sample_set],
    )
    assert_frame_equal(df, df_dup)


@pytest.mark.parametrize(
    "subclass, transcript, sample_sets",
    [
        (
            Ag3,
            "foobar",
            "3.0",
        ),
        (
            Af1,
            "foobar",
            "1.0",
        ),
    ],
)
def test_snp_allele_frequencies__bad_transcript(
    subclass,
    transcript,
    sample_sets,
):
    anoph = setup_subclass_cached(subclass)
    with pytest.raises(ValueError):
        anoph.snp_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
        )


@pytest.mark.parametrize(
    "subclass, cohorts_analysis, transcript, sample_set",
    [
        (
            Ag3,
            "20211101",
            "AGAP004707-RD",
            "AG1000G-FR",
        ),
        (
            Af1,
            "20221129",
            "LOC125767311_t2",
            "1229-VO-GH-DADZIE-VMF00095",
        ),
    ],
)
def test_aa_allele_frequencies__dup_samples(
    subclass, cohorts_analysis, transcript, sample_set
):
    # Expect automatically deduplicate sample sets.
    anoph = setup_subclass_cached(subclass=subclass, cohorts_analysis=cohorts_analysis)
    df = anoph.aa_allele_frequencies(
        transcript=transcript,
        cohorts="admin1_year",
        sample_sets=[sample_set],
    )
    df_dup = anoph.aa_allele_frequencies(
        transcript=transcript,
        cohorts="admin1_year",
        sample_sets=[sample_set, sample_set],
    )
    assert_frame_equal(df, df_dup)


@pytest.mark.parametrize(
    "subclass, cohorts_analysis, transcript, sample_set",
    [
        (
            Ag3,
            "20211101",
            "AGAP004707-RD",
            "AG1000G-FR",
        ),
        (
            Af1,
            "20221129",
            "LOC125767311_t2",
            "1229-VO-GH-DADZIE-VMF00095",
        ),
    ],
)
def test_snp_allele_frequencies_advanced__dup_samples(
    subclass, cohorts_analysis, transcript, sample_set
):
    anoph = setup_subclass_cached(subclass=subclass, cohorts_analysis=cohorts_analysis)
    ds = anoph.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by="admin1_iso",
        period_by="year",
        sample_sets=[sample_set],
    )
    ds_dup = anoph.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by="admin1_iso",
        period_by="year",
        sample_sets=[sample_set, sample_set],
    )
    assert ds.dims == ds_dup.dims


@pytest.mark.parametrize(
    "subclass, cohorts_analysis, transcript, sample_set",
    [
        (
            Ag3,
            "20211101",
            "AGAP004707-RD",
            "AG1000G-FR",
        ),
        (
            Af1,
            "20221129",
            "LOC125767311_t2",
            "1229-VO-GH-DADZIE-VMF00095",
        ),
    ],
)
def test_aa_allele_frequencies_advanced__dup_samples(
    subclass, cohorts_analysis, transcript, sample_set
):
    anoph = setup_subclass_cached(subclass=subclass, cohorts_analysis=cohorts_analysis)
    ds_dup = anoph.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by="admin1_iso",
        period_by="year",
        sample_sets=[sample_set, sample_set],
    )
    ds = anoph.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by="admin1_iso",
        period_by="year",
        sample_sets=[sample_set],
    )
    assert ds.dims == ds_dup.dims


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
    "subclass, sample_sets, region, analysis, cohort_size",
    [
        (Ag3, "AG1000G-BF-B", "3L", "gamb_colu_arab", 10),
        (Af1, "1229-VO-GH-DADZIE-VMF00095", "3RL", "funestus", 10),
    ],
)
def test_haplotypes__cohort_size(subclass, sample_sets, region, analysis, cohort_size):
    # TODO Migrate this test.
    anoph = setup_subclass_cached(subclass)

    ds = anoph.haplotypes(
        region=region,
        sample_sets=sample_sets,
        analysis=analysis,
        cohort_size=cohort_size,
    )
    assert isinstance(ds, xr.Dataset)

    # check fields
    expected_data_vars = {
        "variant_allele",
        "call_genotype",
    }
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "variant_contig",
        "variant_position",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # check dimensions
    assert set(ds.dims) == {"alleles", "ploidy", "samples", "variants"}

    # check dim lengths
    assert ds.dims["samples"] == cohort_size
    assert ds.dims["alleles"] == 2


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
