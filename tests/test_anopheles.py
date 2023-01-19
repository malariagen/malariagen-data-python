import os

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from pandas.testing import assert_frame_equal

from malariagen_data import Af1, Ag3, Region

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
    return subclass(url, **kwargs)


@pytest.mark.parametrize(
    "subclass,url,release,sample_sets_count",
    [
        (Ag3, None, "3.0", 28),
        (Ag3, "gs://vo_agam_release", "3.0", 28),
        (Ag3, "gcs://vo_agam_release", "3.0", 28),
        (Ag3, "simplecache::gs://vo_agam_release/", "3.0", 28),
        (Ag3, "simplecache::gcs://vo_agam_release/", "3.0", 28),
        (Af1, None, "1.0", 8),
        (Af1, "gs://vo_afun_release", "1.0", 8),
        (Af1, "gcs://vo_afun_release", "1.0", 8),
        (Af1, "simplecache::gs://vo_afun_release/", "1.0", 8),
        (Af1, "simplecache::gcs://vo_afun_release/", "1.0", 8),
    ],
)
def test_sample_sets(subclass, url, release, sample_sets_count):

    anoph = setup_subclass(subclass, url)
    df_sample_sets = anoph.sample_sets(release=release)
    assert isinstance(df_sample_sets, pd.DataFrame)
    assert len(df_sample_sets) == sample_sets_count
    assert tuple(df_sample_sets.columns) == ("sample_set", "sample_count", "release")

    # test duplicates not allowed
    with pytest.raises(ValueError):
        anoph.sample_sets(release=[release, release])

    # test default is all public releases
    df_default = anoph.sample_sets()
    df_all = anoph.sample_sets(release=anoph.releases)
    assert_frame_equal(df_default, df_all)


@pytest.mark.parametrize(
    "subclass,url,major_release,major_release_prefix,expected_pre_releases_min",
    [
        (Ag3, "simplecache::gs://vo_agam_release/", "3.0", "3.", 1),
        (Af1, "simplecache::gs://vo_afun_release/", "1.0", "1.", 0),
    ],
)
def test_releases(
    subclass, url, major_release, major_release_prefix, expected_pre_releases_min
):

    anoph = setup_subclass(subclass, url)
    assert isinstance(anoph.releases, tuple)
    assert anoph.releases == (major_release,)

    anoph = setup_subclass(subclass, url, pre=True)
    assert isinstance(anoph.releases, tuple)
    # Note: test_ag3.py has assert len(ag3.releases) > 1 because pre should give > 1 releases
    assert len(anoph.releases) > expected_pre_releases_min
    assert all([r.startswith(major_release_prefix) for r in anoph.releases])


@pytest.mark.parametrize(
    "subclass,url,major_release,sample_set,sample_sets",
    [
        (
            Ag3,
            "simplecache::gs://vo_agam_release/",
            "3.0",
            "AG1000G-X",
            ["AG1000G-BF-A", "AG1000G-BF-B", "AG1000G-BF-C"],
        ),
        (
            Af1,
            "simplecache::gs://vo_afun_release/",
            "1.0",
            "1229-VO-GH-DADZIE-VMF00095",
            [
                "1229-VO-GH-DADZIE-VMF00095",
                "1230-VO-GA-CF-AYALA-VMF00045",
                "1231-VO-MULTI-WONDJI-VMF00043",
            ],
        ),
    ],
)
def test_sample_metadata(subclass, url, major_release, sample_set, sample_sets):

    anoph = setup_subclass(subclass, url)
    df_sample_sets_major = anoph.sample_sets(release=major_release)

    expected_cols = (
        "sample_id",
        "partner_sample_id",
        "contributor",
        "country",
        "location",
        "year",
        "month",
        "latitude",
        "longitude",
        "sex_call",
        "sample_set",
        "release",
    )

    # all major_release
    df_samples_major = anoph.sample_metadata(sample_sets=major_release)
    assert tuple(df_samples_major.columns[: len(expected_cols)]) == expected_cols
    expected_len = df_sample_sets_major["sample_count"].sum()
    assert len(df_samples_major) == expected_len

    # single sample set
    df_samples_single = anoph.sample_metadata(sample_sets=sample_set)
    assert tuple(df_samples_single.columns[: len(expected_cols)]) == expected_cols
    expected_len = df_sample_sets_major.query(f"sample_set == '{sample_set}'")[
        "sample_count"
    ].sum()
    assert len(df_samples_single) == expected_len

    # multiple sample sets
    df_samples_multi = anoph.sample_metadata(sample_sets=sample_sets)
    assert tuple(df_samples_multi.columns[: len(expected_cols)]) == expected_cols
    loc_sample_sets = df_sample_sets_major["sample_set"].isin(sample_sets)
    expected_len = df_sample_sets_major.loc[loc_sample_sets]["sample_count"].sum()
    assert len(df_samples_multi) == expected_len

    # duplicate sample sets
    with pytest.raises(ValueError):
        anoph.sample_metadata(sample_sets=[major_release, major_release])
    with pytest.raises(ValueError):
        # test_ag3.py used AG1000G-UG here instead of AG1000G-X
        anoph.sample_metadata(sample_sets=[sample_set, sample_set])
    with pytest.raises(ValueError):
        anoph.sample_metadata(sample_sets=[sample_set, major_release])

    # default is all public releases
    df_default = anoph.sample_metadata()
    df_all = anoph.sample_metadata(sample_sets=anoph.releases)
    assert_frame_equal(df_default, df_all)


@pytest.mark.parametrize(
    "subclass,mask",
    [(Ag3, "gamb_colu_arab"), (Ag3, "gamb_colu"), (Ag3, "arab"), (Af1, "funestus")],
)
def test_open_site_filters(subclass, mask):
    # check can open the zarr directly
    anoph = setup_subclass(subclass)
    root = anoph.open_site_filters(mask=mask)
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in anoph.contigs:
        assert contig in root


@pytest.mark.parametrize("subclass", [Ag3, Af1])
def test_open_snp_sites(subclass):
    anoph = setup_subclass(subclass)
    root = anoph.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in anoph.contigs:
        assert contig in root


@pytest.mark.parametrize(
    "subclass,sample_set", [(Ag3, "AG1000G-AO"), (Af1, "1229-VO-GH-DADZIE-VMF00095")]
)
def test_open_snp_genotypes(subclass, sample_set):
    # check can open the zarr directly
    anoph = setup_subclass(subclass)
    root = anoph.open_snp_genotypes(sample_set=sample_set)
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in anoph.contigs:
        assert contig in root


@pytest.mark.parametrize("subclass", [Ag3, Af1])
def test_genome(subclass):

    anoph = setup_subclass(subclass)

    # test the open_genome() method to access as zarr
    genome = anoph.open_genome()
    assert isinstance(genome, zarr.hierarchy.Group)
    for contig in anoph.contigs:
        assert contig in genome
        assert genome[contig].dtype == "S1"

    # test the genome_sequence() method to access sequences
    for contig in anoph.contigs:
        seq = anoph.genome_sequence(contig)
        assert isinstance(seq, da.Array)
        assert seq.dtype == "S1"


@pytest.mark.parametrize("subclass", [Ag3, Af1])
def test_sample_metadata_dtypes(subclass):

    expected_dtypes = {
        "sample_id": object,
        "partner_sample_id": object,
        "contributor": object,
        "country": object,
        "location": object,
        "year": "int64",
        "month": "int64",
        "latitude": "float64",
        "longitude": "float64",
        "sex_call": object,
        "sample_set": object,
        "release": object,
    }

    # check all available sample sets
    anoph = setup_subclass(subclass, pre=True)
    df_samples = anoph.sample_metadata()
    for k, v in expected_dtypes.items():
        assert df_samples[k].dtype == v, k

    # check sample sets one by one, just to be sure
    for sample_set in anoph.sample_sets()["sample_set"]:
        df_samples = anoph.sample_metadata(sample_sets=sample_set)
        for k, v in expected_dtypes.items():
            assert df_samples[k].dtype == v, k


@pytest.mark.parametrize("subclass", [Ag3, Af1])
def test_genome_features(subclass):

    anoph = setup_subclass(subclass)

    # default
    df = anoph.genome_features()
    assert isinstance(df, pd.DataFrame)
    gff3_cols = [
        "contig",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
    ]
    if subclass == Af1:
        # different default attributes for funestus
        expected_cols = gff3_cols + ["ID", "Parent", "Note", "description"]
    else:
        expected_cols = gff3_cols + ["ID", "Parent", "Name", "description"]
    assert df.columns.tolist() == expected_cols

    # don't unpack attributes
    df = anoph.genome_features(attributes=None)
    assert isinstance(df, pd.DataFrame)
    expected_cols = gff3_cols + ["attributes"]
    assert df.columns.tolist() == expected_cols


@pytest.mark.parametrize(
    "subclass, region",
    [
        (Ag3, "AGAP007280"),
        (Ag3, "3R:28,000,000-29,000,000"),
        (Ag3, "2R"),
        (Ag3, "X"),
        (Ag3, ["3R", "3L"]),
        (Af1, "3RL:28,000,000-29,000,000"),
        (Af1, "LOC125762289"),
        (Af1, "2RL"),
        (Af1, "X"),
        (Af1, ["2RL", "3RL"]),
    ],
)
def test_genome_features_region(subclass, region):

    anoph = setup_subclass(subclass)

    df = anoph.genome_features(region=region)
    assert isinstance(df, pd.DataFrame)
    gff3_cols = [
        "contig",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
    ]
    if subclass == Af1:
        # different default attributes for funestus
        expected_cols = gff3_cols + ["ID", "Parent", "Note", "description"]
    else:
        expected_cols = gff3_cols + ["ID", "Parent", "Name", "description"]
    assert df.columns.tolist() == expected_cols
    assert len(df) > 0

    # check region
    region = anoph.resolve_region(region)
    if isinstance(region, Region):
        assert np.all(df["contig"].values == region.contig)
        if region.start and region.end:
            assert np.all(df.eval(f"start <= {region.end} and end >= {region.start}"))


@pytest.mark.parametrize("subclass", [Ag3, Af1])
def test_open_site_annotations(subclass):

    anoph = setup_subclass(subclass)

    # test access as zarr
    root = anoph.open_site_annotations()
    assert isinstance(root, zarr.hierarchy.Group)
    for f in (
        "codon_degeneracy",
        "codon_nonsyn",
        "codon_position",
        "seq_cls",
        "seq_flen",
        "seq_relpos_start",
        "seq_relpos_stop",
    ):
        assert f in root
        for contig in anoph.contigs:
            assert contig in root[f]
            z = root[f][contig]
            # raw zarr data is aligned with genome sequence
            assert z.shape == (len(anoph.genome_sequence(region=contig)),)


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

    anoph = setup_subclass(subclass, cohorts_analysis=cohorts_analysis)

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
    df_coh = anoph.sample_cohorts(sample_sets=sample_sets)
    coh_nm = "cohort_" + cohorts
    coh_counts = df_coh[coh_nm].dropna().value_counts().to_frame()
    cohort_labels = coh_counts[coh_counts[coh_nm] >= min_cohort_size].index.to_list()
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
    anoph = setup_subclass(subclass)
    with pytest.raises(ValueError):
        anoph.snp_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=[sample_set, sample_set],
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
    anoph = setup_subclass(subclass=subclass, cohorts_analysis=cohorts_analysis)
    with pytest.raises(ValueError):
        anoph.aa_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=[sample_set, sample_set],
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
def test_snp_allele_frequencies_advanced__dup_samples(
    subclass, cohorts_analysis, transcript, sample_set
):
    anoph = setup_subclass(subclass=subclass, cohorts_analysis=cohorts_analysis)
    with pytest.raises(ValueError):
        anoph.snp_allele_frequencies_advanced(
            transcript=transcript,
            area_by="admin1_iso",
            period_by="year",
            sample_sets=[sample_set, sample_set],
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
def test_aa_allele_frequencies_advanced__dup_samples(
    subclass, cohorts_analysis, transcript, sample_set
):
    anoph = setup_subclass(subclass=subclass, cohorts_analysis=cohorts_analysis)
    with pytest.raises(ValueError):
        anoph.aa_allele_frequencies_advanced(
            transcript=transcript,
            area_by="admin1_iso",
            period_by="year",
            sample_sets=[sample_set, sample_set],
        )


@pytest.mark.parametrize(
    "subclass, sample_sets",
    [
        (
            Ag3,
            "3.0",
        ),
        (
            Af1,
            "1.0",
        ),
    ],
)
def test_sample_metadata_with_cohorts(subclass, sample_sets):
    anoph = setup_subclass(subclass)
    df_samples_coh = anoph.sample_metadata(sample_sets=sample_sets)
    for c in expected_cohort_cols:
        assert c in df_samples_coh


@pytest.mark.parametrize(
    "subclass, sample_sets, test_subdir",
    [
        (
            Ag3,
            "3.0",
            "ag",
        ),
        (
            Af1,
            "1.0",
            "af",
        ),
    ],
)
def test_sample_metadata_without_cohorts(subclass, sample_sets, test_subdir):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_path = os.path.join(
        working_dir, "anopheles_test_data", "test_missing_cohorts", test_subdir
    )
    anoph = setup_subclass(subclass, url=test_data_path)
    df_samples_coh = anoph.sample_metadata(sample_sets=sample_sets)
    for c in expected_cohort_cols:
        assert c in df_samples_coh
        assert df_samples_coh[c].isnull().all()
