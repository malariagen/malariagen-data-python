import random

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr  # type: ignore
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
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
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
    *,
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
    n_samples = ds.sizes["samples"]
    assert n_samples == n_samples_expected
    if min_cohort_size:
        assert n_samples >= min_cohort_size
    assert ds.sizes["ploidy"] == 2
    assert ds.sizes["alleles"] == 2

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


# check behaviour when no haplotype data is present within a sample set
def test_haplotypes_with_empty_calls(ag3_sim_fixture, ag3_sim_api: AnophelesHapData):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fix a sample set that will be empty for the fixed (arab) analysis calls
    sample_set = "AG1000G-AO"
    region = fixture.random_region_str()
    analysis = "arab"

    check_haplotypes(
        fixture=fixture,
        api=api,
        sample_sets=sample_set,
        region=region,
        analysis=analysis,
        sample_query=None,
    )


@pytest.mark.parametrize("analysis", ["arab", "gamb_colu", "gamb_colu_arab"])
@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_haplotypes_virtual_contigs(
    ag3_sim_fixture,
    ag3_sim_api: AnophelesHapData,
    analysis,
    chrom,
):
    fixture = ag3_sim_fixture
    api = ag3_sim_api
    contig_r, contig_l = api.virtual_contigs[chrom]

    # Standard checks, whole chromosome.
    check_haplotypes(
        fixture=fixture, api=api, region=chrom, analysis=analysis, sample_sets=None
    )

    try:
        api.haplotypes(region=contig_r, analysis=analysis)

    except ValueError:
        # Assume no haplotypes available for the given analysis.
        with pytest.raises(ValueError):
            api.haplotypes(region=contig_l, analysis=analysis)
        with pytest.raises(ValueError):
            api.haplotypes(region=chrom, analysis=analysis)

    else:
        # Extra checks, whole chromosome.
        ds_r = api.haplotypes(region=contig_r, analysis=analysis)
        ds_l = api.haplotypes(region=contig_l, analysis=analysis)
        ds_chrom = api.haplotypes(region=chrom, analysis=analysis)
        assert isinstance(ds_chrom, xr.Dataset)
        assert len(ds_chrom.dims) == 4
        assert ds_chrom.sizes["variants"] == (
            ds_r.sizes["variants"] + ds_l.sizes["variants"]
        )
        for dim in "samples", "alleles", "ploidy":
            assert ds_chrom.sizes[dim] == ds_r.sizes[dim] == ds_l.sizes[dim]
        assert ds_chrom["call_genotype"].dtype == "int8"
        assert ds_chrom["variant_position"].dtype == "int32"
        pos = ds_chrom["variant_position"].values
        assert np.all(pos[1:] > pos[:-1])  # monotonically increasing

        # Test with region.
        seq = api.genome_sequence(region=chrom)
        start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
        region = f"{chrom}:{start:,}-{stop:,}"

        # Standard checks.
        check_haplotypes(
            fixture=fixture, api=api, region=region, analysis=analysis, sample_sets=None
        )

        # Extra checks.
        ds_region = api.haplotypes(region=region, analysis=analysis)
        assert isinstance(ds_region, xr.Dataset)
        assert len(ds_region.dims) == 4
        for dim in "samples", "alleles", "ploidy":
            assert ds_region.sizes[dim] == ds_chrom.sizes[dim]
        assert ds_region["call_genotype"].dtype == "int8"
        assert ds_region["variant_position"].dtype == "int32"
        pos = ds_region["variant_position"].values
        assert np.all(pos[1:] > pos[:-1])  # monotonically increasing
        assert np.all(pos >= start)
        assert np.all(pos <= stop)


def check_haplotype_sites(*, api: AnophelesHapData, region):
    pos = api.haplotype_sites(region=region, field="POS")
    ref = api.haplotype_sites(region=region, field="REF")
    alt = api.haplotype_sites(region=region, field="ALT")
    assert isinstance(pos, da.Array)
    assert pos.ndim == 1
    assert pos.dtype == "i4"
    assert isinstance(ref, da.Array)
    assert ref.ndim == 1
    assert ref.dtype == "S1"
    assert isinstance(alt, da.Array)
    assert alt.ndim == 1
    assert alt.dtype == "S1"
    assert pos.shape[0] == ref.shape[0] == alt.shape[0]


@parametrize_with_cases("fixture,api", cases=".")
def test_haplotype_sites(fixture, api: AnophelesHapData):
    # Test with contig.
    contig = fixture.random_contig()
    check_haplotype_sites(api=api, region=contig)

    # Test with region string.
    region = fixture.random_region_str()
    check_haplotype_sites(api=api, region=region)

    # Test with genome feature ID.
    df_gff = api.genome_features(attributes=["ID"])
    region = random.choice(df_gff["ID"].dropna().to_list())
    check_haplotype_sites(api=api, region=region)


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_haplotype_sites_with_virtual_contigs(ag3_sim_api, chrom):
    api = ag3_sim_api

    # Standard checks.
    check_haplotype_sites(api=api, region=chrom)

    # Extra checks.
    contig_r, contig_l = api.virtual_contigs[chrom]
    pos_r = api.haplotype_sites(region=contig_r, field="POS")
    pos_l = api.haplotype_sites(region=contig_l, field="POS")
    offset = api.genome_sequence(region=contig_r).shape[0]
    pos_expected = da.concatenate([pos_r, pos_l + offset])
    pos_actual = api.haplotype_sites(region=chrom, field="POS")
    assert da.all(pos_expected == pos_actual).compute(scheduler="single-threaded")

    # Test with region.
    seq = api.genome_sequence(region=chrom)
    start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
    region = f"{chrom}:{start:,}-{stop:,}"

    # Standard checks.
    check_haplotype_sites(api=api, region=region)

    # Extra checks.
    region_size = stop - start
    pos = api.haplotype_sites(region=region, field="POS").compute()
    assert pos.shape[0] <= region_size
    assert np.all(pos >= start)
    assert np.all(pos <= stop)
