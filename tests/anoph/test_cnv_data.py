import random

import bokeh.models
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.cnv_data import AnophelesCnvData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesCnvData(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
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
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        default_coverage_calls_analysis="gamb_colu",
        discordant_read_calls_analysis=ag3_sim_fixture.config[
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS"
        ],
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesCnvData(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        default_coverage_calls_analysis="funestus",
        discordant_read_calls_analysis=None,
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


@parametrize_with_cases("fixture,api", cases=".")
def test_open_cnv_hmm(fixture, api: AnophelesCnvData):
    for rec in api.sample_sets().itertuples():
        sample_set = rec.sample_set
        root = api.open_cnv_hmm(sample_set=sample_set)
        assert isinstance(root, zarr.hierarchy.Group)
        assert "sample_coverage_variance" in root
        assert "sample_is_high_variance" in root
        assert "samples" in root
        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]
            assert "variants" in contig_grp
            assert "calldata" in contig_grp
            assert "samples" in contig_grp
            variants_grp = contig_grp["variants"]
            assert "END" in variants_grp
            assert "POS" in variants_grp
            calldata_grp = contig_grp["calldata"]
            assert "CN" in calldata_grp
            assert "NormCov" in calldata_grp
            assert "RawCov" in calldata_grp

    # Check with a sample set that should not exist
    with pytest.raises(ValueError):
        root = api.open_cnv_hmm(sample_set="foobar")


@parametrize_with_cases("fixture,api", cases=".")
def test_open_cnv_coverage_calls(fixture, api: AnophelesCnvData):
    for analysis in api.coverage_calls_analysis_ids:
        for rec in api.sample_sets().itertuples():
            sample_set = rec.sample_set
            root = api.open_cnv_coverage_calls(sample_set=sample_set, analysis=analysis)
            assert isinstance(root, zarr.hierarchy.Group)
            assert "samples" in root
            for contig in api.contigs:
                assert contig in root
                contig_grp = root[contig]
                assert "variants" in contig_grp
                assert "calldata" in contig_grp
                assert "samples" in contig_grp
                variants_grp = contig_grp["variants"]
                assert "CIEND" in variants_grp
                assert "CIPOS" in variants_grp
                assert "END" in variants_grp
                assert "FILTER_PASS" in variants_grp
                assert "FILTER_qMerge" in variants_grp
                assert "ID" in variants_grp
                assert "POS" in variants_grp
                calldata_grp = contig_grp["calldata"]
                assert "GT" in calldata_grp

    # Check with a sample set that should not exist
    with pytest.raises(ValueError):
        root = api.open_cnv_coverage_calls(
            sample_set="foobar", analysis=random.choice(api.coverage_calls_analysis_ids)
        )

    # Check with an analysis that should not exist
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    with pytest.raises(ValueError):
        root = api.open_cnv_coverage_calls(
            sample_set=random.choice(all_sample_sets), analysis="foobar"
        )

    # Check with a sample set and analysis that should not exist
    with pytest.raises(ValueError):
        root = api.open_cnv_coverage_calls(sample_set="foobar", analysis="bazqux")


@parametrize_with_cases("fixture,api", cases=".")
def test_open_cnv_discordant_read_calls(fixture, api: AnophelesCnvData):
    for rec in api.sample_sets().itertuples():
        sample_set = rec.sample_set
        root = api.open_cnv_discordant_read_calls(sample_set=sample_set)
        assert isinstance(root, zarr.hierarchy.Group)
        assert "sample_coverage_variance" in root
        assert "sample_is_high_variance" in root
        assert "samples" in root
        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]
            assert "variants" in contig_grp
            assert "calldata" in contig_grp
            assert "samples" in contig_grp
            variants_grp = contig_grp["variants"]
            assert "END" in variants_grp
            assert "EndBreakpointMethod" in variants_grp
            assert "ID" in variants_grp
            assert "POS" in variants_grp
            assert "Region" in variants_grp
            assert "StartBreakpointMethod" in variants_grp
            calldata_grp = contig_grp["calldata"]
            assert "GT" in calldata_grp

    # Check with a sample set that should not exist
    with pytest.raises(ValueError):
        root = api.open_cnv_discordant_read_calls(sample_set="foobar")


def test_cnv_hmm__sample_query(ag3_sim_fixture, ag3_sim_api: AnophelesCnvData):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    sample_sets = "AG1000G-BF-A"
    region = fixture.random_region_str()

    # Parametrize query.
    parametrize_query = [
        "taxon == 'coluzzii' and location == 'Bana Village'",
        "taxon == 'gambiae' and location == 'Pala'",
        "taxon == 'robot'",
    ]

    # Run tests.
    for sample_query in parametrize_query:
        # Get the number of samples for this query
        df_samples = ag3_sim_api.sample_metadata().query(sample_query)

        if len(df_samples) == 0:
            with pytest.raises(ValueError):
                ds = api.cnv_hmm(
                    region=region,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    max_coverage_variance=None,
                )
        else:
            ds = api.cnv_hmm(
                region=region,
                sample_sets=sample_sets,
                sample_query=sample_query,
                max_coverage_variance=None,
            )
            assert isinstance(ds, xr.Dataset)

            # check fields
            expected_data_vars = {
                "call_CN",
                "call_NormCov",
                "call_RawCov",
                "sample_coverage_variance",
                "sample_is_high_variance",
            }
            assert set(ds.data_vars) == expected_data_vars

            expected_coords = {
                "variant_contig",
                "variant_position",
                "variant_end",
                "sample_id",
            }
            assert set(ds.coords) == expected_coords

            # check dimensions
            assert set(ds.dims) == {"samples", "variants"}

            # check expected samples
            df_samples = api.sample_metadata(sample_sets=sample_sets).query(
                sample_query
            )
            expected_samples = df_samples["sample_id"].tolist()
            n_samples_expected = len(expected_samples)
            assert ds.sizes["samples"] == n_samples_expected

            # check sample IDs
            assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()


def test_cnv_hmm__sample_query_options(ag3_sim_fixture, ag3_sim_api: AnophelesCnvData):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    sample_sets = "AG1000G-BF-A"
    region = fixture.random_region_str()

    # Query options.
    taxon_list = ["coluzzii", "gambiae"]
    location_list = ["Bana Village", "Pala"]
    non_taxon_list = ["robot", "cyborg"]
    sample_query_options = {
        "local_dict": {
            "taxon_list": taxon_list,
            "location_list": location_list,
            "non_taxon_list": non_taxon_list,
        }
    }

    # Parametrize query.
    parametrize_query = [
        "taxon in @taxon_list and location in @location_list",
        "taxon in @non_taxon_list",
    ]

    # Run tests.
    for sample_query in parametrize_query:
        # Get the number of samples for this query
        df_samples = ag3_sim_api.sample_metadata().query(
            sample_query, **sample_query_options
        )

        if len(df_samples) == 0:
            with pytest.raises(ValueError):
                ds = api.cnv_hmm(
                    region=region,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                    max_coverage_variance=None,
                )
        else:
            ds = api.cnv_hmm(
                region=region,
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
                max_coverage_variance=None,
            )
            assert isinstance(ds, xr.Dataset)

            # check fields
            expected_data_vars = {
                "call_CN",
                "call_NormCov",
                "call_RawCov",
                "sample_coverage_variance",
                "sample_is_high_variance",
            }
            assert set(ds.data_vars) == expected_data_vars

            expected_coords = {
                "variant_contig",
                "variant_position",
                "variant_end",
                "sample_id",
            }
            assert set(ds.coords) == expected_coords

            # check dimensions
            assert set(ds.dims) == {"samples", "variants"}

            # check expected samples
            df_samples = api.sample_metadata(sample_sets=sample_sets).query(
                sample_query, **sample_query_options
            )
            expected_samples = df_samples["sample_id"].tolist()
            n_samples_expected = len(expected_samples)
            assert ds.sizes["samples"] == n_samples_expected

            # check sample IDs
            assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm(fixture, api: AnophelesCnvData):
    # Parametrize sample_sets.
    all_releases = api.releases
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Parametrize region.
    parametrize_region = [
        fixture.random_contig(),
        random.sample(api.contigs, 2),
        fixture.random_region_str(),
    ]

    for sample_sets in parametrize_sample_sets:
        for region in parametrize_region:
            ds = api.cnv_hmm(
                region=region, sample_sets=sample_sets, max_coverage_variance=None
            )
            assert isinstance(ds, xr.Dataset)

            # check fields
            expected_data_vars = {
                "call_CN",
                "call_NormCov",
                "call_RawCov",
                "sample_coverage_variance",
                "sample_is_high_variance",
            }
            assert set(ds.data_vars) == expected_data_vars

            expected_coords = {
                "variant_contig",
                "variant_position",
                "variant_end",
                "sample_id",
            }
            assert set(ds.coords) == expected_coords

            # check dimensions
            assert set(ds.dims) == {"samples", "variants"}

            # check dim lengths
            df_samples = api.sample_metadata(sample_sets=sample_sets)
            n_samples_expected = len(df_samples)
            assert ds.sizes["samples"] == n_samples_expected

            # check sample IDs
            assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()

            # check shapes
            for f in expected_coords | expected_data_vars:
                x = ds[f]
                assert isinstance(x, xr.DataArray)
                assert isinstance(x.data, da.Array)
                if f.startswith("variant_"):
                    assert x.ndim == 1
                    assert x.dims == ("variants",)
                elif f.startswith("call_"):
                    assert x.ndim == 2
                    assert x.dims == ("variants", "samples")
                elif f.startswith("sample_"):
                    assert x.ndim == 1
                    assert x.dims == ("samples",)

            # check attributes
            assert "contigs" in ds.attrs
            assert ds.attrs["contigs"] == api.contigs

            # check can set up computations
            d1 = ds["variant_position"] > 10_000
            assert isinstance(d1, xr.DataArray)
            d2 = ds["call_CN"].sum(axis=1)
            assert isinstance(d2, xr.DataArray)


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm__max_coverage_variance(fixture, api: AnophelesCnvData):
    # Set up test.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    region = fixture.random_contig()

    # Parametrize max_coverage_variance.
    parametrize_max_coverage_variance = np.random.uniform(low=0, high=1, size=4)

    for max_coverage_variance in parametrize_max_coverage_variance:
        ds = api.cnv_hmm(
            region=region,
            sample_sets=sample_set,
            max_coverage_variance=max_coverage_variance,
        )
        assert isinstance(ds, xr.Dataset)

        # check fields
        expected_data_vars = {
            "call_CN",
            "call_NormCov",
            "call_RawCov",
            "sample_coverage_variance",
            "sample_is_high_variance",
        }
        assert set(ds.data_vars) == expected_data_vars

        expected_coords = {
            "variant_contig",
            "variant_position",
            "variant_end",
            "sample_id",
        }
        assert set(ds.coords) == expected_coords

        # check dimensions
        assert set(ds.dims) == {"samples", "variants"}

        # check expected samples
        cov_var = ds["sample_coverage_variance"].values
        assert np.all(cov_var <= max_coverage_variance)


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_coverage_calls(fixture, api: AnophelesCnvData):
    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = random.sample(all_sample_sets, 3)

    # Parametrize analysis.
    parametrize_analysis = api.coverage_calls_analysis_ids

    # Parametrize region.
    parametrize_region = [
        fixture.random_contig(),
        random.sample(api.contigs, 2),
        fixture.random_region_str(),
    ]

    # Reduce nested indentation by using a generator.
    param_generator = (
        (sample_set, analysis, region)
        for sample_set in parametrize_sample_sets
        for analysis in parametrize_analysis
        for region in parametrize_region
    )

    for sample_set, analysis, region in param_generator:
        ds = api.cnv_coverage_calls(
            region=region, analysis=analysis, sample_set=sample_set
        )
        assert isinstance(ds, xr.Dataset)

        # check fields
        expected_data_vars = {
            "variant_CIPOS",
            "variant_CIEND",
            "variant_filter_pass",
            "call_genotype",
        }
        assert set(ds.data_vars) == expected_data_vars

        expected_coords = {
            "variant_contig",
            "variant_position",
            "variant_end",
            "variant_id",
            "sample_id",
        }
        assert set(ds.coords) == expected_coords

        # check dimensions
        assert set(ds.dims) == {"samples", "variants"}

        # check sample IDs
        df_samples = api.sample_metadata(sample_sets=sample_set)
        sample_id = pd.Series(ds["sample_id"].values)
        assert sample_id.isin(df_samples["sample_id"]).all()

        # check shapes
        for f in expected_coords | expected_data_vars:
            x = ds[f]
            assert isinstance(x, xr.DataArray)
            assert isinstance(x.data, da.Array)

            if f.startswith("variant_"):
                assert x.ndim == 1
                assert x.dims == ("variants",)
            elif f.startswith("call_"):
                assert x.ndim == 2
                assert x.dims == ("variants", "samples")
            elif f.startswith("sample_"):
                assert x.ndim == 1
                assert x.dims == ("samples",)

        # check attributes
        assert "contigs" in ds.attrs
        assert ds.attrs["contigs"] == api.contigs

        # check can set up computations
        d1 = ds["variant_position"] > 10_000
        assert isinstance(d1, xr.DataArray)
        d2 = ds["call_genotype"].sum(axis=1)
        assert isinstance(d2, xr.DataArray)


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_discordant_read_calls(fixture, api: AnophelesCnvData):
    # Parametrize sample_sets.
    all_releases = api.releases
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Parametrize contig.
    parametrize_contig = [
        random.choice(api.contigs),
        random.sample(api.contigs, 2),
    ]

    for sample_sets in parametrize_sample_sets:
        for contig in parametrize_contig:
            ds = api.cnv_discordant_read_calls(contig=contig, sample_sets=sample_sets)
            assert isinstance(ds, xr.Dataset)

            # check fields
            expected_data_vars = {
                "variant_Region",
                "variant_StartBreakpointMethod",
                "variant_EndBreakpointMethod",
                "call_genotype",
                "sample_coverage_variance",
                "sample_is_high_variance",
            }
            assert set(ds.data_vars) == expected_data_vars

            expected_coords = {
                "variant_contig",
                "variant_position",
                "variant_end",
                "variant_id",
                "sample_id",
            }
            assert set(ds.coords) == expected_coords

            # check dimensions
            assert set(ds.dims) == {"samples", "variants"}

            # check dim lengths
            df_samples = api.sample_metadata(sample_sets=sample_sets)
            n_samples = len(df_samples)
            assert ds.sizes["samples"] == n_samples

            # check sample IDs
            assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()

            # check shapes
            for f in expected_coords | expected_data_vars:
                x = ds[f]
                assert isinstance(x, xr.DataArray)
                assert isinstance(x.data, da.Array)

                if f.startswith("variant_"):
                    assert x.ndim == 1
                    assert x.dims == ("variants",)
                elif f.startswith("call_"):
                    assert x.ndim == 2
                    assert x.dims == ("variants", "samples")
                elif f.startswith("sample_"):
                    assert x.ndim == 1
                    assert x.dims == ("samples",)
                    assert x.shape == (n_samples,)

            # check attributes
            assert "contigs" in ds.attrs
            assert ds.attrs["contigs"] == api.contigs

            # check can set up computations
            d1 = ds["variant_position"] > 10_000
            assert isinstance(d1, xr.DataArray)
            d2 = ds["call_genotype"].sum(axis=1)
            assert isinstance(d2, xr.DataArray)

    # Check with a contig that should not exist
    with pytest.raises(ValueError):
        api.cnv_discordant_read_calls(
            contig="foobar", sample_sets=random.choice(all_sample_sets)
        )

    # Check with a sample set that should not exist
    with pytest.raises(ValueError):
        api.cnv_discordant_read_calls(
            contig=random.choice(api.contigs), sample_sets="foobar"
        )

    # Check with a contig and sample set that should not exist
    with pytest.raises(ValueError):
        api.cnv_discordant_read_calls(contig="foobar", sample_sets="bazqux")


def test_cnv_discordant_read_calls__sample_query(
    ag3_sim_fixture, ag3_sim_api: AnophelesCnvData
):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    sample_sets = "AG1000G-BF-A"
    contig = fixture.random_contig()

    # Parametrize query.
    parametrize_query = [
        "taxon == 'coluzzii' and location == 'Bana Village'",
        "taxon == 'gambiae' and location == 'Pala'",
        "taxon == 'robot'",
    ]

    # Run tests.
    for sample_query in parametrize_query:
        # Get the number of samples for this query
        df_samples = ag3_sim_api.sample_metadata().query(sample_query)

        if len(df_samples) == 0:
            with pytest.raises(ValueError):
                ds = api.cnv_discordant_read_calls(
                    contig=contig,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                )
        else:
            ds = api.cnv_discordant_read_calls(
                contig=contig,
                sample_sets=sample_sets,
                sample_query=sample_query,
            )
            assert isinstance(ds, xr.Dataset)

            # check fields
            expected_data_vars = {
                "variant_Region",
                "variant_StartBreakpointMethod",
                "variant_EndBreakpointMethod",
                "call_genotype",
                "sample_coverage_variance",
                "sample_is_high_variance",
            }
            assert set(ds.data_vars) == expected_data_vars

            expected_coords = {
                "variant_position",
                "variant_end",
                "variant_id",
                "variant_contig",
                "sample_id",
            }
            assert set(ds.coords) == expected_coords

            # check dimensions
            assert set(ds.dims) == {"variants", "samples"}

            # check expected samples
            df_samples = api.sample_metadata(sample_sets=sample_sets).query(
                sample_query
            )
            expected_samples = df_samples["sample_id"].tolist()
            n_samples_expected = len(expected_samples)
            assert ds.sizes["samples"] == n_samples_expected

            # check sample IDs
            assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()


def test_cnv_discordant_read_calls__sample_query_options(
    ag3_sim_fixture, ag3_sim_api: AnophelesCnvData
):
    api = ag3_sim_api
    fixture = ag3_sim_fixture

    # Fixed parameters.
    sample_sets = "AG1000G-BF-A"
    contig = fixture.random_contig()

    # Query options.
    taxon_list = ["coluzzii", "gambiae"]
    location_list = ["Bana Village", "Pala"]
    non_taxon_list = ["robot", "cyborg"]
    sample_query_options = {
        "local_dict": {
            "taxon_list": taxon_list,
            "location_list": location_list,
            "non_taxon_list": non_taxon_list,
        }
    }

    # Parametrize query.
    parametrize_query = [
        "taxon in @taxon_list and location in @location_list",
        "taxon in @non_taxon_list",
    ]

    # Run tests.
    for sample_query in parametrize_query:
        # Get the number of samples for this query
        df_samples = ag3_sim_api.sample_metadata().query(
            sample_query, **sample_query_options
        )

        if len(df_samples) == 0:
            with pytest.raises(ValueError):
                ds = api.cnv_discordant_read_calls(
                    contig=contig,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                )
        else:
            ds = api.cnv_discordant_read_calls(
                contig=contig,
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )
            assert isinstance(ds, xr.Dataset)

            # check fields
            expected_data_vars = {
                "variant_Region",
                "variant_StartBreakpointMethod",
                "variant_EndBreakpointMethod",
                "call_genotype",
                "sample_coverage_variance",
                "sample_is_high_variance",
            }
            assert set(ds.data_vars) == expected_data_vars

            expected_coords = {
                "variant_position",
                "variant_end",
                "variant_id",
                "variant_contig",
                "sample_id",
            }
            assert set(ds.coords) == expected_coords

            # check dimensions
            assert set(ds.dims) == {"variants", "samples"}

            # check expected samples
            df_samples = api.sample_metadata(sample_sets=sample_sets).query(
                sample_query,
                **sample_query_options,
            )
            expected_samples = df_samples["sample_id"].tolist()
            n_samples_expected = len(expected_samples)
            assert ds.sizes["samples"] == n_samples_expected

            # check sample IDs
            assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_cnv_hmm_coverage_track(fixture, api: AnophelesCnvData):
    # Set up test.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    region = fixture.random_contig()
    df_samples = api.sample_metadata(sample_sets=sample_set)
    all_sample_ids = df_samples["sample_id"].values
    sample_id = np.random.choice(all_sample_ids)

    fig = api.plot_cnv_hmm_coverage_track(
        sample=sample_id,
        region=region,
        show=False,
    )

    # Check return type.
    assert isinstance(fig, bokeh.model.Model)

    # Check non-default y_max
    fig2 = api.plot_cnv_hmm_coverage_track(
        sample=sample_id, region=region, show=False, y_max=12.3
    )

    # Check return type.
    assert isinstance(fig2, bokeh.model.Model)

    # Check with a sample that should not exist.
    # Note: this currently raises KeyError rather than ValueError
    with pytest.raises(KeyError):
        api.plot_cnv_hmm_coverage_track(
            sample="foo",
            region=region,
            show=False,
        )

    # Check with a region that should not exist.
    with pytest.raises(ValueError):
        api.plot_cnv_hmm_coverage_track(
            sample=sample_id,
            region="foo",
            show=False,
        )

    # Check with a sample and region that should not exist.
    with pytest.raises(ValueError):
        api.plot_cnv_hmm_coverage_track(
            sample="foo",
            region="bar",
            show=False,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_cnv_hmm_coverage(fixture, api: AnophelesCnvData):
    # Set up test.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    region = fixture.random_contig()
    df_samples = api.sample_metadata(sample_sets=sample_set)
    all_sample_ids = df_samples["sample_id"].values
    sample_id = np.random.choice(all_sample_ids)

    fig = api.plot_cnv_hmm_coverage(
        sample=sample_id,
        region=region,
        show=False,
    )

    # Check return type.
    assert isinstance(fig, bokeh.model.Model)

    # Check with a sample that should not exist.
    # Note: this currently raises KeyError rather than ValueError
    with pytest.raises(KeyError):
        api.plot_cnv_hmm_coverage(
            sample="foo",
            region=region,
            show=False,
        )

    # Check with a region that should not exist.
    with pytest.raises(ValueError):
        api.plot_cnv_hmm_coverage(
            sample=sample_id,
            region="foo",
            show=False,
        )

    # Check with a sample and region that should not exist.
    with pytest.raises(ValueError):
        api.plot_cnv_hmm_coverage(
            sample="foo",
            region="bar",
            show=False,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_cnv_hmm_heatmap_track(fixture, api: AnophelesCnvData):
    # Parametrize region (not regions).
    parametrize_region = [
        fixture.random_contig(),
        fixture.random_region_str(),
    ]

    # Parametrize sample_sets.
    all_releases = api.releases
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    for region in parametrize_region:
        for sample_sets in parametrize_sample_sets:
            fig = api.plot_cnv_hmm_heatmap_track(
                region=region,
                sample_sets=sample_sets,
                show=False,
            )

            # Check return type.
            assert isinstance(fig, bokeh.model.Model)

    # Check with a non-default sample_query.
    fig2 = api.plot_cnv_hmm_heatmap_track(
        region=fixture.random_contig(),
        # Note: assuming this query will always return samples.
        sample_query="sex_call == 'F'",
        show=False,
    )
    assert isinstance(fig2, bokeh.model.Model)

    # Check with a non-default height.
    fig3 = api.plot_cnv_hmm_heatmap_track(
        region=fixture.random_contig(),
        height=1,
        show=False,
    )
    assert isinstance(fig3, bokeh.model.Model)

    # Check with a region that should not exist.
    with pytest.raises(ValueError):
        api.plot_cnv_hmm_heatmap_track(
            region="foo",
            show=False,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_cnv_hmm_heatmap(fixture, api: AnophelesCnvData):
    # Parametrize region (not regions).
    parametrize_region = [
        fixture.random_contig(),
        fixture.random_region_str(),
    ]

    for region in parametrize_region:
        fig = api.plot_cnv_hmm_heatmap(
            region=region,
            show=False,
        )

        # Check return type.
        assert isinstance(fig, bokeh.model.Model)

    # Check with a region that should not exist
    with pytest.raises(ValueError):
        api.plot_cnv_hmm_heatmap(
            region="foo",
        )
