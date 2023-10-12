import random

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.hap_data import AnophelesHapData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesHapData(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
        gff_gene_type="gene",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        default_phasing_analysis="gamb_colu_arab",
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesHapData(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        default_phasing_analysis="funestus",
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


def _check_haplotype_sites(root, api: AnophelesHapData):
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in api.contigs:
        assert contig in root
        contig_grp = root[contig]
        assert "variants" in contig_grp
        variants = contig_grp["variants"]
        assert "POS" in variants
        assert "REF" in variants
        assert "ALT" in variants


@parametrize_with_cases("fixture,api", cases=".")
def test_open_haplotype_sites(fixture, api: AnophelesHapData):
    # Test default analysis.
    root = api.open_haplotype_sites()
    _check_haplotype_sites(root, api)

    # Test specific analyses.
    for analysis in api.phasing_analysis_ids:
        root = api.open_haplotype_sites(analysis=analysis)
        _check_haplotype_sites(root, api)

        # Test _haplotype_sites_for_contig().
        for contig in api.contigs:
            haplotype_pos = api._haplotype_sites_for_contig(
                contig=contig,
                analysis=analysis,
                field="POS",
                inline_array=True,
                chunks="native",
            ).compute()
            assert len(haplotype_pos) == len(root[contig]["variants"]["POS"])


@parametrize_with_cases("fixture,api", cases=".")
def test_open_haplotypes(fixture, api: AnophelesHapData):
    for rec in api.sample_sets().itertuples():
        sample_set = rec.sample_set
        for analysis in api.phasing_analysis_ids:
            # How many samples do we expect?
            expected_samples = fixture.phasing_samples[sample_set, analysis]

            # How many sites do we expect?
            expected_n_sites = fixture.n_hap_sites[analysis]

            # Access haplotype data.
            root = api.open_haplotypes(sample_set=sample_set, analysis=analysis)
            if len(expected_samples) == 0:
                assert root is None
            else:
                assert isinstance(root, zarr.hierarchy.Group)

                # Check samples array.
                assert "samples" in root
                samples = root["samples"][:]
                assert samples.ndim == 1
                assert samples.dtype.kind == "O"
                assert samples.shape[0] == len(expected_samples)

                # Check calldata arrays.
                for contig in api.contigs:
                    assert contig in root
                    contig_grp = root[contig]

                    assert "calldata" in contig_grp
                    calldata = contig_grp["calldata"]
                    assert "GT" in calldata
                    gt = calldata["GT"]
                    assert gt.ndim == 3
                    assert gt.dtype == "i1"
                    assert gt.shape[0] == expected_n_sites[contig]
                    assert gt.shape[1] == len(expected_samples)
                    assert gt.shape[2] == 2


def check_haplotypes(
    fixture,
    api: AnophelesHapData,
    sample_sets,
    region,
    analysis,
    sample_query=None,
    cohort_size=None,
    min_cohort_size=None,
    max_cohort_size=None,
):
    # Set up test, figure out how many samples phased in the analysis.
    sample_sets_prepped = api._prep_sample_sets_param(sample_sets=sample_sets)
    samples_phased = np.concatenate(
        [
            fixture.phasing_samples[sample_set, analysis]
            for sample_set in sample_sets_prepped
        ]
    )
    n_samples_phased = len(samples_phased)

    # Check if no samples phased in the analysis.
    if n_samples_phased == 0:
        with pytest.raises(ValueError):
            ds = api.haplotypes(
                region=region,
                sample_sets=sample_sets,
                analysis=analysis,
                sample_query=sample_query,
            )
        return

    # Handle sample query.
    if sample_query is not None:
        df_samples = api.sample_metadata(sample_sets=sample_sets)
        df_samples = df_samples.set_index("sample_id")
        df_samples_phased = df_samples.loc[samples_phased].reset_index()
        df_samples_queried = df_samples_phased.query(sample_query)
        samples_selected = df_samples_queried["sample_id"].values
    else:
        samples_selected = samples_phased
    n_samples_selected = len(samples_selected)

    # Check if no samples matching selection.
    if n_samples_selected == 0:
        with pytest.raises(ValueError):
            ds = api.haplotypes(
                region=region,
                sample_sets=sample_sets,
                analysis=analysis,
                sample_query=sample_query,
            )
        return

    # Check if not enough samples for requested cohort size.
    if cohort_size and n_samples_selected < cohort_size:
        with pytest.raises(ValueError):
            ds = api.haplotypes(
                region=region,
                sample_sets=sample_sets,
                analysis=analysis,
                sample_query=sample_query,
                cohort_size=cohort_size,
            )
        return

    # Check if not enough samples for requested minimum cohort size.
    if min_cohort_size and n_samples_selected < min_cohort_size:
        with pytest.raises(ValueError):
            ds = api.haplotypes(
                region=region,
                sample_sets=sample_sets,
                analysis=analysis,
                sample_query=sample_query,
                min_cohort_size=min_cohort_size,
            )
        return

    # Call function to be tested.
    ds = api.haplotypes(
        region=region,
        sample_sets=sample_sets,
        analysis=analysis,
        sample_query=sample_query,
        cohort_size=cohort_size,
        min_cohort_size=min_cohort_size,
        max_cohort_size=max_cohort_size,
    )

    # Check return type.
    assert isinstance(ds, xr.Dataset)

    # Check variables.
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

    # Check dimensions.
    assert set(ds.dims) == {"alleles", "ploidy", "samples", "variants"}

    # Check samples.
    samples = ds["sample_id"].values
    if cohort_size or max_cohort_size:
        # N.B., there may have been some down-sampling.
        selected_samples_set = set(samples_selected)
        assert all([s in selected_samples_set for s in samples])
    else:
        assert set(samples) == set(samples_selected)

    # Check dim lengths.
    if cohort_size:
        n_samples_expected = cohort_size
    elif max_cohort_size:
        n_samples_expected = min(n_samples_selected, max_cohort_size)
    else:
        n_samples_expected = n_samples_selected
    n_samples = ds.dims["samples"]
    assert n_samples == n_samples_expected
    if min_cohort_size:
        assert n_samples >= min_cohort_size
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 2

    # Check shapes.
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert x.ndim == 2
            assert x.shape[1] == 2
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim == 1
            assert x.dims == ("variants",)
        elif f == "call_genotype":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape[1] == n_samples_expected
            assert x.shape[2] == 2

    # Check attributes.
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == api.contigs

    # Check can set up computations.
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_genotype"].sum(axis=(1, 2))
    assert isinstance(d2, xr.DataArray)


@parametrize_with_cases("fixture,api", cases=".")
def test_haplotypes_with_sample_sets_param(fixture, api: AnophelesHapData):
    # Fixed parameters.
    region = fixture.random_region_str()
    analysis = api.phasing_analysis_ids[0]

    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_releases = api.releases
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Run tests.
    for sample_sets in parametrize_sample_sets:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_haplotypes_with_region_param(fixture, api: AnophelesHapData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    analysis = api.phasing_analysis_ids[0]

    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        [fixture.random_region_str(), fixture.random_region_str()],
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Run tests.
    for region in parametrize_region:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_haplotypes_with_analysis_param(fixture, api: AnophelesHapData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize analysis.
    parametrize_analysis = api.phasing_analysis_ids

    # Run tests.
    for analysis in parametrize_analysis:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
        )


def test_haplotypes_with_sample_query_param(
    ag3_sim_fixture, ag3_sim_api: AnophelesHapData
):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    analysis = api.phasing_analysis_ids[0]

    # Parametrize analysis.
    parametrize_query = ["sex_call == 'F'", "taxon == 'coluzzii'", "taxon == 'robot'"]

    # Run tests.
    for sample_query in parametrize_query:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
            sample_query=sample_query,
        )


def test_haplotypes_with_cohort_size_param(
    ag3_sim_fixture, ag3_sim_api: AnophelesHapData
):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    analysis = api.phasing_analysis_ids[0]

    # Parametrize over cohort_size.
    parametrize_cohort_size = [random.randint(1, 10), random.randint(10, 50), 1_000]
    for cohort_size in parametrize_cohort_size:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
            sample_query=None,
            cohort_size=cohort_size,
        )


def test_haplotypes_with_min_cohort_size_param(
    ag3_sim_fixture, ag3_sim_api: AnophelesHapData
):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    analysis = api.phasing_analysis_ids[0]

    # Parametrize over min_cohort_size.
    parametrize_min_cohort_size = [
        random.randint(1, 10),
        random.randint(10, 50),
        1_000,
    ]
    for min_cohort_size in parametrize_min_cohort_size:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
            sample_query=None,
            min_cohort_size=min_cohort_size,
        )


def test_haplotypes_with_max_cohort_size_param(
    ag3_sim_fixture, ag3_sim_api: AnophelesHapData
):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    analysis = api.phasing_analysis_ids[0]

    # Parametrize over max_cohort_size.
    parametrize_max_cohort_size = [
        random.randint(1, 10),
        random.randint(10, 50),
        1_000,
    ]
    for max_cohort_size in parametrize_max_cohort_size:
        check_haplotypes(
            fixture=fixture,
            api=api,
            sample_sets=sample_sets,
            region=region,
            analysis=analysis,
            sample_query=None,
            max_cohort_size=max_cohort_size,
        )
