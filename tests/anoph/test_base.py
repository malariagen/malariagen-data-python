import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1
from malariagen_data import amin1 as _amin1
from malariagen_data.anoph.base import AnophelesBase
from malariagen_data.anoph.snp_data import AnophelesSnpData
from malariagen_data.util import Region, _locate_region, _resolve_region


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesBase(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesBase(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


@pytest.fixture
def adir1_sim_api(adir1_sim_fixture):
    return AnophelesBase(
        url=adir1_sim_fixture.url,
        public_url=adir1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
    )


@pytest.fixture
def amin1_sim_api(amin1_sim_fixture):
    return AnophelesBase(
        url=amin1_sim_fixture.url,
        public_url=amin1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
    )


@pytest.fixture
def amin1_sim_api_snp(amin1_sim_fixture):
    """Fixture for amin1 with SNP data capabilities, used for tests requiring snp_sites."""
    return AnophelesSnpData(
        url=amin1_sim_fixture.url,
        public_url=amin1_sim_fixture.url,
        config_path=_amin1.CONFIG_PATH,
        major_version_number=_amin1.MAJOR_VERSION_NUMBER,
        major_version_path=_amin1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="minimus",
        results_cache=amin1_sim_fixture.results_cache_path.as_posix(),
    )


# N.B., here we use pytest_cases to parametrize tests. Each
# function whose name begins with "case_" defines a set of
# inputs to the test functions. See the documentation for
# pytest_cases for more information, e.g.:
#
# https://smarie.github.io/python-pytest-cases/#basic-usage
#
# We use this approach here because we want to use fixtures
# as test parameters, which is otherwise hard to do with
# pytest alone.


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


def case_amin1_sim(amin1_sim_fixture, amin1_sim_api):
    return amin1_sim_fixture, amin1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_config(fixture, api):
    config = api.config
    assert isinstance(config, dict)
    assert config == fixture.config


@parametrize_with_cases("fixture,api", cases=".")
def test_releases(fixture, api):
    releases = api.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])
    assert releases == fixture.releases


@parametrize_with_cases("fixture,api", cases=".")
def test_client_location(fixture, api):
    location = api.client_location
    assert isinstance(location, str)


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_sets_default(fixture, api):
    df = api.sample_sets()
    releases = api.releases
    expected = pd.concat(
        [fixture.release_manifests[release] for release in releases],
        axis=0,
        ignore_index=True,
    )
    expected.reset_index(inplace=True, drop=True)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == [
        "sample_set",
        "sample_count",
        "study_id",
        "study_url",
        "terms_of_use_expiry_date",
        "terms_of_use_url",
        "release",
        "unrestricted_use",
    ]
    assert len(df) > 0
    assert_frame_equal(
        df[
            [
                "sample_set",
                "sample_count",
                "study_id",
                "study_url",
                "terms_of_use_expiry_date",
                "terms_of_use_url",
            ]
        ],
        expected,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_sets_release(fixture, api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        assert isinstance(df_ss, pd.DataFrame)
        assert df_ss.columns.tolist() == [
            "sample_set",
            "sample_count",
            "study_id",
            "study_url",
            "terms_of_use_expiry_date",
            "terms_of_use_url",
            "release",
            "unrestricted_use",
        ]
        assert len(df_ss) > 0
        expected = fixture.release_manifests[release]
        assert_frame_equal(
            df_ss[
                [
                    "sample_set",
                    "sample_count",
                    "study_id",
                    "study_url",
                    "terms_of_use_expiry_date",
                    "terms_of_use_url",
                ]
            ],
            expected,
        )
        assert (df_ss["release"] == release).all()

    with pytest.raises(TypeError):
        api.sample_sets(release=3.1)  # type: ignore


@parametrize_with_cases("fixture,api", cases=".")
def test_lookup_release(fixture, api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        for s in df_ss["sample_set"]:
            assert api.lookup_release(s) == release


def test_prep_sample_sets_param(ag3_sim_api: AnophelesBase):
    assert ag3_sim_api._prep_sample_sets_param(sample_sets="3.0") == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets="3.1") == [
        "1177-VO-ML-LEHMANN-VMF00004",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets=["3.0", "3.1"]) == [
        "1177-VO-ML-LEHMANN-VMF00004",
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets=None) == [
        "1177-VO-ML-LEHMANN-VMF00004",
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets="AG1000G-AO") == [
        "AG1000G-AO"
    ]
    assert ag3_sim_api._prep_sample_sets_param(
        sample_sets=["AG1000G-AO", "AG1000G-BF-A"]
    ) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(
        sample_sets=("AG1000G-AO", "AG1000G-BF-A")
    ) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(
        sample_sets=["AG1000G-AO", "AG1000G-BF-A", "AG1000G-AO"]
    ) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    assert ag3_sim_api._prep_sample_sets_param(sample_sets=["3.0", "AG1000G-AO"]) == [
        "AG1000G-AO",
        "AG1000G-BF-A",
    ]
    with pytest.raises(ValueError):
        ag3_sim_api._prep_sample_sets_param(sample_sets=["AG1000G-AO", "foobar"])


@parametrize_with_cases("fixture,api", cases=".")
def test_lookup_study(fixture, api):
    # Set up test.
    df_sample_sets = api.sample_sets()
    all_sample_sets = df_sample_sets["sample_set"].values
    sample_set = np.random.choice(all_sample_sets)

    study_rec_by_sample_set = api.lookup_study(sample_set)
    df_sample_set = df_sample_sets.set_index("sample_set").loc[sample_set]

    # Check we get the same study_id back.
    assert df_sample_set["study_id"] == study_rec_by_sample_set

    # Check we get a study_id string.
    assert isinstance(study_rec_by_sample_set, str)

    with pytest.raises(ValueError):
        api.lookup_study("foobar")


def test_locate_region(amin1_sim_fixture, amin1_sim_api_snp):
    """Test _locate_region and _resolve_region utility functions with amin1 data.

    This test was migrated from tests/integration/test_amin1.py::test_locate_region.
    """
    api = amin1_sim_api_snp

    # Get a contig and gene from the simulated data
    contig = amin1_sim_fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    gene_ids = df_gff["ID"].dropna().to_list()

    if not gene_ids:
        pytest.skip("No gene IDs available in simulated data")

    gene_id = gene_ids[0]
    gene = df_gff.query(f"ID == '{gene_id}'").squeeze()
    gene_start = gene["start"]
    gene_end = gene["end"]
    gene_contig = gene["contig"]

    # Create test regions - using simulated data values
    region_tests = [
        contig,  # Just contig name
        f"{gene_contig}:{gene_start}-{gene_end}",  # Region string
        f"{gene_contig}:{gene_start:,}-{gene_end:,}",  # Region string with commas
        gene_id,  # Gene ID
        Region(gene_contig, gene_start, gene_end),  # Region object
    ]

    for region_raw in region_tests:
        region = _resolve_region(api, region_raw)
        pos = api.snp_sites(region=region.contig, field="POS")
        loc_region = _locate_region(region, pos.compute())

        # Check types
        assert isinstance(loc_region, slice)
        assert isinstance(region, Region)

        # Check Region with contig only
        if region_raw == contig:
            assert region.contig == contig
            assert region.start is None
            assert region.end is None

        # Check that Region object goes through unchanged
        if isinstance(region_raw, Region):
            assert region == region_raw

        # Check that gene name matches coordinates from genome_features
        if region_raw == gene_id:
            assert region == Region(gene_contig, gene_start, gene_end)
            pos_computed = pos.compute()
            if len(pos_computed) > 0:
                # Check if slice is non-empty (start != stop)
                sliced_pos = pos_computed[loc_region]
                if len(sliced_pos) > 0:
                    # Only check if there are SNPs in the region
                    assert sliced_pos[0] >= gene_start
                    assert sliced_pos[-1] <= gene_end

        # Check string parsing with coordinates
        if isinstance(region_raw, str) and ":" in region_raw and "-" in region_raw:
            # Parse the region string to extract coordinates
            parts = region_raw.split(":")
            if len(parts) == 2:
                coord_part = parts[1]
                if "-" in coord_part:
                    start_str, end_str = coord_part.split("-")
                    start = int(start_str.replace(",", ""))
                    end = int(end_str.replace(",", ""))
                    assert region == Region(gene_contig, start, end)
