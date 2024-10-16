import random
from itertools import product

import allel  # type: ignore
import bokeh.model
import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr  # type: ignore
from numpy.testing import assert_array_equal
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.base_params import DEFAULT
from malariagen_data.anoph.snp_data import AnophelesSnpData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSnpData(
        url=ag3_sim_fixture.url,
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
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSnpData(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="funestus",
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
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
def test_open_snp_sites(fixture, api: AnophelesSnpData):
    root = api.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in api.contigs:
        assert contig in root
        contig_grp = root[contig]
        assert "variants" in contig_grp
        variants = contig_grp["variants"]
        assert "POS" in variants
        assert "REF" in variants
        assert "ALT" in variants


def test_site_mask_ids_ag3(ag3_sim_api: AnophelesSnpData):
    assert ag3_sim_api.site_mask_ids == ("gamb_colu_arab", "gamb_colu", "arab")


def test_site_mask_ids_af1(af1_sim_api: AnophelesSnpData):
    assert af1_sim_api.site_mask_ids == ("funestus",)


@parametrize_with_cases("fixture,api", cases=".")
def test_open_site_filters(fixture, api: AnophelesSnpData):
    for mask in api.site_mask_ids:
        root = api.open_site_filters(mask=mask)
        assert isinstance(root, zarr.hierarchy.Group)
        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]
            assert "variants" in contig_grp
            variants_grp = contig_grp["variants"]
            assert "filter_pass" in variants_grp
            filter_pass = variants_grp["filter_pass"]
            assert filter_pass.dtype == bool


@parametrize_with_cases("fixture,api", cases=".")
def test_open_snp_genotypes(fixture, api: AnophelesSnpData):
    for rec in api.sample_sets().itertuples():
        sample_set = rec.sample_set
        n_samples = rec.sample_count
        root = api.open_snp_genotypes(sample_set=sample_set)
        assert isinstance(root, zarr.hierarchy.Group)

        # Check samples array.
        assert "samples" in root
        samples = root["samples"][:]
        assert samples.ndim == 1
        assert samples.shape[0] == n_samples
        assert samples.dtype.kind == "S"

        # Check calldata arrays.
        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]

            n_sites = fixture.n_snp_sites[contig]
            assert "calldata" in contig_grp
            calldata = contig_grp["calldata"]
            assert "GT" in calldata
            gt = calldata["GT"]
            assert gt.shape == (n_sites, n_samples, 2)
            assert gt.dtype == "i1"
            assert "GQ" in calldata
            gq = calldata["GQ"]
            assert gq.shape == (n_sites, n_samples)
            assert gq.dtype == "i1"
            assert "MQ" in calldata
            mq = calldata["MQ"]
            assert mq.shape == (n_sites, n_samples)
            assert mq.dtype == "f4"
            assert "AD" in calldata
            ad = calldata["AD"]
            assert ad.shape == (n_sites, n_samples, 4)
            assert ad.dtype == "i2"


def check_site_filters(api: AnophelesSnpData, mask, region):
    filter_pass = api.site_filters(region=region, mask=mask)
    assert isinstance(filter_pass, da.Array)
    assert filter_pass.ndim == 1
    assert filter_pass.dtype == bool


@parametrize_with_cases("fixture,api", cases=".")
def test_site_filters(fixture, api: AnophelesSnpData):
    for mask in api.site_mask_ids:
        # Test with contig.
        contig = fixture.random_contig()
        check_site_filters(api, mask=mask, region=contig)

        # Test with region string.
        region = fixture.random_region_str()
        check_site_filters(api, mask=mask, region=region)

        # Test with genome feature ID.
        df_gff = api.genome_features(attributes=["ID"])
        region = random.choice(df_gff["ID"].dropna().to_list())
        check_site_filters(api, mask=mask, region=region)


def check_snp_sites(api: AnophelesSnpData, region):
    pos = api.snp_sites(region=region, field="POS")
    ref = api.snp_sites(region=region, field="REF")
    alt = api.snp_sites(region=region, field="ALT")
    assert isinstance(pos, da.Array)
    assert pos.ndim == 1
    assert pos.dtype == "i4"
    assert isinstance(ref, da.Array)
    assert ref.ndim == 1
    assert ref.dtype == "S1"
    assert isinstance(alt, da.Array)
    assert alt.ndim == 2
    assert alt.dtype == "S1"
    assert pos.shape[0] == ref.shape[0] == alt.shape[0]

    # Apply site mask.
    mask = random.choice(api.site_mask_ids)
    filter_pass = api.site_filters(region=region, mask=mask).compute()
    n_pass = np.count_nonzero(filter_pass)
    pos_pass = api.snp_sites(
        region=region,
        field="POS",
        site_mask=mask,
    )
    assert isinstance(pos_pass, da.Array)
    assert pos_pass.ndim == 1
    assert pos_pass.dtype == "i4"
    assert pos_pass.shape[0] == n_pass
    assert pos_pass.compute().shape == pos_pass.shape
    for f in "POS", "REF", "ALT":
        d = api.snp_sites(
            region=region,
            site_mask=mask,
            field=f,
        )
        assert isinstance(d, da.Array)
        assert d.shape[0] == n_pass
        assert d.shape == d.compute().shape


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_sites(fixture, api: AnophelesSnpData):
    # Test with contig.
    contig = fixture.random_contig()
    check_snp_sites(api=api, region=contig)

    # Test with region string.
    region = fixture.random_region_str()
    check_snp_sites(api=api, region=region)

    # Test with genome feature ID.
    df_gff = api.genome_features(attributes=["ID"])
    region = random.choice(df_gff["ID"].dropna().to_list())
    check_snp_sites(api=api, region=region)


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_sites_with_virtual_contigs(ag3_sim_api, chrom):
    api = ag3_sim_api

    # Standard checks.
    check_snp_sites(api, region=chrom)

    # Extra checks.
    contig_r, contig_l = api.virtual_contigs[chrom]
    pos_r = api.snp_sites(region=contig_r, field="POS")
    pos_l = api.snp_sites(region=contig_l, field="POS")
    offset = api.genome_sequence(region=contig_r).shape[0]
    pos_expected = da.concatenate([pos_r, pos_l + offset])
    pos_actual = api.snp_sites(region=chrom, field="POS")
    assert da.all(pos_expected == pos_actual).compute(scheduler="single-threaded")

    # Test with region.
    seq = api.genome_sequence(region=chrom)
    start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
    region = f"{chrom}:{start:,}-{stop:,}"

    # Standard checks.
    check_snp_sites(api, region=region)

    # Extra checks.
    region_size = stop - start
    pos = api.snp_sites(region=region, field="POS").compute()
    assert pos.shape[0] <= region_size
    assert np.all(pos >= start)
    assert np.all(pos <= stop)


@parametrize_with_cases("fixture,api", cases=".")
def test_open_site_annotations(fixture, api):
    root = api.open_site_annotations()
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
        for contig in api.contigs:
            assert contig in root[f]
            z = root[f][contig]
            # Zarr data should be aligned with genome sequence.
            assert z.shape == (len(api.genome_sequence(region=contig)),)


def _check_site_annotations(api: AnophelesSnpData, region, site_mask):
    ds_snp = api.snp_variants(region=region, site_mask=site_mask)
    n_variants = ds_snp.sizes["variants"]
    ds_ann = api.site_annotations(region=region, site_mask=site_mask)
    # Site annotations dataset should be aligned with SNP sites.
    assert ds_ann.sizes["variants"] == n_variants
    assert isinstance(ds_ann, xr.Dataset)
    for f in (
        "codon_degeneracy",
        "codon_nonsyn",
        "codon_position",
        "seq_cls",
        "seq_flen",
        "seq_relpos_start",
        "seq_relpos_stop",
    ):
        d = ds_ann[f]
        assert d.ndim == 1
        assert d.dims == ("variants",)
        assert d.shape == (n_variants,)


@parametrize_with_cases("fixture,api", cases=".")
def test_site_annotations(fixture, api):
    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    # Don't need to support multiple regions at this time.
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Parametrize site_mask.
    parametrize_site_mask = (None, random.choice(api.site_mask_ids))

    # Run tests.
    for region, site_mask in product(
        parametrize_region,
        parametrize_site_mask,
    ):
        _check_site_annotations(
            api=api,
            region=region,
            site_mask=site_mask,
        )


def check_snp_genotypes(
    api, region, sample_sets=None, sample_query=None, sample_query_options={}
):
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )

    # Check default field (GT).
    default = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    assert isinstance(default, da.Array)
    assert default.ndim == 3
    assert default.dtype == "i1"
    assert default.shape[1] == len(df_samples)

    # Check GT.
    gt = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        field="GT",
    )
    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert gt.shape == default.shape

    # Check GQ.
    gq = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        field="GQ",
    )
    assert isinstance(gq, da.Array)
    assert gq.ndim == 2
    assert gq.dtype == "i1"
    assert gq.shape == gt.shape[:2]

    # Check MQ.
    mq = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        field="MQ",
    )
    assert isinstance(mq, da.Array)
    assert mq.ndim == 2
    assert mq.dtype == "f4"
    assert mq.shape == gt.shape[:2]

    # Check AD.
    ad = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        field="AD",
    )
    assert isinstance(ad, da.Array)
    assert ad.ndim == 3
    assert ad.dtype == "i2"
    assert ad.shape[:2] == gt.shape[:2]
    assert ad.shape[2] == 4

    # Check with site mask.
    mask = random.choice(api.site_mask_ids)
    filter_pass = api.site_filters(region=region, mask=mask).compute()
    gt_pass = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        site_mask=mask,
    )
    assert isinstance(gt_pass, da.Array)
    assert gt_pass.ndim == 3
    assert gt_pass.dtype == "i1"
    assert gt_pass.shape[0] == np.count_nonzero(filter_pass)
    assert gt_pass.shape[1] == len(df_samples)
    assert gt_pass.shape[2] == 2

    # Check native versus auto chunks.
    gt_native = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        chunks="native",
    )
    gt_auto = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        chunks="auto",
    )
    assert gt_native.chunks != gt_auto.chunks


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_genotypes_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()

    # Parametrize sample_sets.
    all_releases = api.releases
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Run tests.
    for sample_sets in parametrize_sample_sets:
        check_snp_genotypes(api=api, sample_sets=sample_sets, region=region)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_genotypes_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)

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
        check_snp_genotypes(api=api, sample_sets=sample_sets, region=region)


@pytest.mark.parametrize(
    "sample_query",
    ["sex_call == 'F'", "taxon == 'coluzzii'", "taxon == 'robot'"],
)
def test_snp_genotypes_with_sample_query_param(
    ag3_sim_api: AnophelesSnpData, sample_query
):
    contig = random.choice(ag3_sim_api.contigs)
    df_samples = ag3_sim_api.sample_metadata().query(sample_query)

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.snp_genotypes(region=contig, sample_query=sample_query)

    else:
        check_snp_genotypes(api=ag3_sim_api, region=contig, sample_query=sample_query)


@pytest.mark.parametrize(
    "sample_query,sample_query_options",
    [
        pytest.param(
            "sex_call in @sex_call_list", {"local_dict": {"sex_call_list": ["F", "M"]}}
        ),
        pytest.param(
            "taxon in @taxon_list",
            {"local_dict": {"taxon_list": ["coluzzii", "arabiensis"]}},
        ),
        pytest.param(
            "taxon in @taxon_list", {"local_dict": {"taxon_list": ["robot", "cyborg"]}}
        ),
    ],
)
def test_snp_genotypes_with_sample_query_options_param(
    ag3_sim_api: AnophelesSnpData, sample_query, sample_query_options
):
    contig = random.choice(ag3_sim_api.contigs)
    df_samples = ag3_sim_api.sample_metadata().query(
        sample_query, **sample_query_options
    )

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.snp_genotypes(
                region=contig,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )

    else:
        check_snp_genotypes(
            api=ag3_sim_api,
            region=contig,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_genotypes_with_virtual_contigs(ag3_sim_api, chrom):
    api = ag3_sim_api

    # Standard checks.
    check_snp_genotypes(api, region=chrom)

    # Extra checks.
    contig_r, contig_l = api.virtual_contigs[chrom]
    d_r = api.snp_genotypes(region=contig_r)
    d_l = api.snp_genotypes(region=contig_l)
    d = da.concatenate([d_r, d_l])
    gt = api.snp_genotypes(region=chrom)
    assert gt.shape == d.shape

    # Test with region.
    seq = api.genome_sequence(region=chrom)
    start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
    region = f"{chrom}:{start:,}-{stop:,}"
    # Standard checks.
    check_snp_genotypes(api, region=region)
    # Extra checks.
    pos = api.snp_sites(region=region, field="POS")
    gt = api.snp_genotypes(region=region)
    assert pos.shape[0] == gt.shape[0]


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_variants_with_virtual_contigs(ag3_sim_api, chrom):
    api = ag3_sim_api

    # Test with whole chromosome.
    pos = api.snp_sites(region=chrom, field="POS").compute()
    ds_chrom = api.snp_variants(region=chrom)
    assert isinstance(ds_chrom, xr.Dataset)
    assert len(ds_chrom.dims) == 2
    assert ds_chrom.sizes["variants"] == pos.shape[0]
    assert ds_chrom["variant_position"].dtype == "int32"
    assert_array_equal(pos, ds_chrom["variant_position"].values)

    # Test with region.
    seq = api.genome_sequence(region=chrom)
    start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
    region = f"{chrom}:{start:,}-{stop:,}"
    pos = api.snp_sites(region=region, field="POS").compute()
    ds_region = api.snp_variants(region=region)
    assert isinstance(ds_region, xr.Dataset)
    assert len(ds_region.dims) == 2
    assert ds_region.sizes["variants"] == pos.shape[0]
    assert ds_region["variant_position"].dtype == "int32"
    assert_array_equal(pos, ds_region["variant_position"].values)


def check_snp_calls(api, sample_sets, region, site_mask):
    ds = api.snp_calls(region=region, sample_sets=sample_sets, site_mask=site_mask)
    assert isinstance(ds, xr.Dataset)

    # Check fields.
    expected_data_vars = {
        "variant_allele",
        "call_genotype",
        "call_genotype_mask",
        "call_GQ",
        "call_AD",
        "call_MQ",
    }
    for m in api.site_mask_ids:
        expected_data_vars.add(f"variant_filter_pass_{m}")
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "variant_contig",
        "variant_position",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # Check dimensions.
    assert set(ds.dims) == {"alleles", "ploidy", "samples", "variants"}

    # Check dim lengths.
    pos = api.snp_sites(region=region, field="POS", site_mask=site_mask)
    n_variants = len(pos)
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    assert ds.sizes["variants"] == n_variants
    assert ds.sizes["samples"] == n_samples
    assert ds.sizes["ploidy"] == 2
    assert ds.sizes["alleles"] == 4

    # Check shapes.
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert x.ndim == 2
            assert x.shape == (n_variants, 4)
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim == 1
            assert x.shape == (n_variants,)
            assert x.dims == ("variants",)
        elif f in {"call_genotype", "call_genotype_mask"}:
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape == (n_variants, n_samples, 2)
        elif f == "call_AD":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "alleles")
            assert x.shape == (n_variants, n_samples, 4)
        elif f.startswith("call_"):
            assert x.ndim == 2
            assert x.dims == ("variants", "samples")
            assert x.shape == (n_variants, n_samples)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # Check samples.
    expected_samples = df_samples["sample_id"].tolist()
    assert ds["sample_id"].values.tolist() == expected_samples

    # Check attributes.
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == api.contigs

    # Check can set up computations.
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xr.DataArray)

    # Check compress bug.
    pos = ds["variant_position"].data
    assert pos.shape == pos.compute().shape


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

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
        check_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

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
        check_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_site_mask_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize site_mask.
    parametrize_site_mask = (None, DEFAULT) + api.site_mask_ids

    # Run tests.
    for site_mask in parametrize_site_mask:
        check_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@pytest.mark.parametrize(
    "sample_query",
    ["sex_call == 'F'", "taxon == 'coluzzii'", "taxon == 'robot'"],
)
def test_snp_calls_with_sample_query_param(ag3_sim_api: AnophelesSnpData, sample_query):
    df_samples = ag3_sim_api.sample_metadata().query(sample_query)

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.snp_calls(region="3L", sample_query=sample_query)

    else:
        ds = ag3_sim_api.snp_calls(region="3L", sample_query=sample_query)
        assert ds.sizes["samples"] == len(df_samples)
        assert_array_equal(ds["sample_id"].values, df_samples["sample_id"].values)


@pytest.mark.parametrize(
    "sample_query,sample_query_options",
    [
        pytest.param(
            "sex_call in @sex_call_list", {"local_dict": {"sex_call_list": ["F", "M"]}}
        ),
        pytest.param(
            "taxon in @taxon_list",
            {"local_dict": {"taxon_list": ["coluzzii", "arabiensis"]}},
        ),
        pytest.param(
            "taxon in @taxon_list", {"local_dict": {"taxon_list": ["robot", "cyborg"]}}
        ),
    ],
)
def test_snp_calls_with_sample_query_options_param(
    ag3_sim_api: AnophelesSnpData, sample_query, sample_query_options
):
    df_samples = ag3_sim_api.sample_metadata().query(
        sample_query, **sample_query_options
    )

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.snp_calls(
                region="3L",
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )

    else:
        ds = ag3_sim_api.snp_calls(
            region="3L",
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )
        assert ds.sizes["samples"] == len(df_samples)
        assert_array_equal(ds["sample_id"].values, df_samples["sample_id"].values)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_min_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with minimum cohort size.
    ds = api.snp_calls(
        sample_sets=sample_sets,
        region=region,
        min_cohort_size=10,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["samples"] >= 10
    with pytest.raises(ValueError):
        api.snp_calls(
            sample_sets=sample_sets,
            region=region,
            min_cohort_size=1_000,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_max_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with maximum cohort size.
    ds = api.snp_calls(
        sample_sets=sample_sets,
        region=region,
        max_cohort_size=15,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["samples"] <= 15


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with specific cohort size.
    cohort_size = random.randint(1, 10)
    ds = api.snp_calls(
        sample_sets=sample_sets,
        region=region,
        cohort_size=cohort_size,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["samples"] == cohort_size
    with pytest.raises(ValueError):
        api.snp_calls(
            sample_sets=sample_sets,
            region=region,
            cohort_size=1_000,
        )


@pytest.mark.parametrize(
    "site_class",
    [
        "CDS_DEG_4",
        "CDS_DEG_2_SIMPLE",
        "CDS_DEG_0",
        "INTRON_SHORT",
        "INTRON_LONG",
        "INTRON_SPLICE_5PRIME",
        "INTRON_SPLICE_3PRIME",
        "UTR_5PRIME",
        "UTR_3PRIME",
        "INTERGENIC",
    ],
)
def test_snp_calls_with_site_class_param(ag3_sim_api: AnophelesSnpData, site_class):
    ds1 = ag3_sim_api.snp_calls(region="3L")
    ds2 = ag3_sim_api.snp_calls(region="3L", site_class=site_class)
    assert ds2.sizes["variants"] < ds1.sizes["variants"]


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_calls_with_virtual_contigs(ag3_sim_api, chrom):
    api = ag3_sim_api

    # Test with whole chromosome.

    # Standard checks.
    check_snp_calls(api, region=chrom, sample_sets=None, site_mask=None)

    # Extra checks.
    pos = api.snp_sites(region=chrom, field="POS").compute()
    ds_chrom = api.snp_calls(region=chrom)
    assert isinstance(ds_chrom, xr.Dataset)
    assert len(ds_chrom.dims) == 4
    assert pos.shape[0] == ds_chrom.sizes["variants"]
    assert pos.shape[0] == ds_chrom["call_genotype"].shape[0]
    assert ds_chrom["call_genotype"].dtype == "int8"
    assert ds_chrom["variant_position"].dtype == "int32"
    assert_array_equal(pos, ds_chrom["variant_position"].values)

    # Test with region.
    seq = api.genome_sequence(region=chrom)
    start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
    region = f"{chrom}:{start:,}-{stop:,}"

    # Standard checks.
    check_snp_calls(api, region=region, sample_sets=None, site_mask=None)

    # Extra checks.
    ds_region = api.snp_calls(region=region)
    pos = api.snp_sites(region=region, field="POS")
    assert isinstance(ds_region, xr.Dataset)
    assert len(ds_region.dims) == 4
    assert ds_region.sizes["samples"] == ds_chrom.sizes["samples"]
    assert pos.shape[0] == ds_region.sizes["variants"]
    assert pos.shape[0] == ds_region["call_genotype"].shape[0]
    assert ds_region["call_genotype"].dtype == "int8"
    assert ds_region["variant_position"].dtype == "int32"
    assert_array_equal(pos, ds_region["variant_position"].values)


def check_snp_allele_counts(
    *,
    api,
    region,
    sample_sets,
    sample_query,
    sample_query_options=None,
    site_mask,
):
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    n_samples = len(df_samples)

    # Run once to compute results.
    ac = api.snp_allele_counts(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        site_mask=site_mask,
    )
    assert isinstance(ac, np.ndarray)
    pos = api.snp_sites(region=region, field="POS", site_mask=site_mask)
    assert ac.shape == (pos.shape[0], 4)
    assert np.all(ac >= 0)
    an = ac.sum(axis=1)
    assert an.max() <= 2 * n_samples

    # Run again to ensure loading from results cache produces the same result.
    ac2 = api.snp_allele_counts(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        site_mask=site_mask,
    )
    assert_array_equal(ac, ac2)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

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
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=None,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

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
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=None,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_site_mask_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize site_mask.
    parametrize_site_mask = (None,) + api.site_mask_ids

    # Run tests.
    for site_mask in parametrize_site_mask:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=None,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_sample_query_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize sample_query.
    parametrize_sample_query = (None, "sex_call == 'F'")

    # Run tests.
    for sample_query in parametrize_sample_query:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=sample_query,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_sample_query_options_param(
    fixture, api: AnophelesSnpData
):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)
    sample_query_options = {
        "local_dict": {
            "sex_call_list": ["F", "M"],
        }
    }

    # Parametrize sample_query.
    parametrize_sample_query = (None, "sex_call in @sex_call_list")

    # Run tests.
    for sample_query in parametrize_sample_query:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )


def _check_is_accessible(api: AnophelesSnpData, region, mask):
    is_accessible = api.is_accessible(region=region, site_mask=mask)
    assert isinstance(is_accessible, np.ndarray)
    assert is_accessible.ndim == 1
    assert is_accessible.shape[0] == api.genome_sequence(region=region).shape[0]


@parametrize_with_cases("fixture,api", cases=".")
def test_is_accessible(fixture, api: AnophelesSnpData):
    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    # Don't need to support multiple regions at this time.
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Parametrize site_mask.
    parametrize_site_mask = api.site_mask_ids

    # Run tests.
    for region, site_mask in product(
        parametrize_region,
        parametrize_site_mask,
    ):
        _check_is_accessible(
            api=api,
            region=region,
            mask=site_mask,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_snps(fixture, api: AnophelesSnpData):
    # Randomly choose parameter values.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    site_mask = random.choice(api.site_mask_ids)

    # Exercise the function.
    fig = api.plot_snps(
        region=region,
        sample_sets=sample_sets,
        site_mask=site_mask,
        show=False,
    )
    assert isinstance(fig, bokeh.model.Model)


def check_biallelic_snp_calls_and_diplotypes(
    api,
    region,
    sample_sets=None,
    site_mask=None,
    site_class=None,
    min_minor_ac=None,
    max_missing_an=None,
    n_snps=None,
):
    ds = api.biallelic_snp_calls(
        region=region,
        sample_sets=sample_sets,
        site_mask=site_mask,
        site_class=site_class,
        min_minor_ac=min_minor_ac,
        max_missing_an=max_missing_an,
        n_snps=n_snps,
    )
    assert isinstance(ds, xr.Dataset)

    # Check fields.
    expected_data_vars = {
        "variant_allele",
        "variant_allele_count",
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

    # Check dim lengths.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    n_variants = ds.sizes["variants"]
    # assert n_variants > 0
    assert ds.sizes["samples"] == n_samples
    assert ds.sizes["ploidy"] == 2
    assert ds.sizes["alleles"] == 2

    # Check shapes.
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        if f == "variant_allele_count":
            assert isinstance(x.data, np.ndarray)
        else:
            assert isinstance(x.data, da.Array)
        assert isinstance(x.values, np.ndarray)

        if f.startswith("variant_allele"):
            assert x.ndim == 2
            assert x.shape == (n_variants, 2)
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim == 1
            assert x.shape == (n_variants,)
            assert x.dims == ("variants",)
        elif f == "call_genotype":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape == (n_variants, n_samples, 2)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # Check samples.
    expected_samples = df_samples["sample_id"].tolist()
    assert ds["sample_id"].values.tolist() == expected_samples

    # Check attributes.
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == api.contigs

    # Check can set up computations.
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)

    # Check if any variants found, could be zero.
    if ds.sizes["variants"] == 0:
        # Bail out early, can't run further tests.
        return ds

    # Check biallelic genotypes.
    gt = ds["call_genotype"].data
    assert gt.compute().max() <= 1
    assert gt.max().compute() <= 1

    # Check compress bug.
    pos = ds["variant_position"].data
    assert pos.shape == pos.compute().shape

    # Check computation of diplotypes.
    gn, samples = api.biallelic_diplotypes(
        region=region,
        sample_sets=sample_sets,
        site_mask=site_mask,
        site_class=site_class,
        min_minor_ac=min_minor_ac,
        max_missing_an=max_missing_an,
        n_snps=n_snps,
    )
    assert isinstance(gn, np.ndarray)
    assert isinstance(samples, np.ndarray)
    assert gn.ndim == 2
    assert gn.shape[0] == ds.sizes["variants"]
    assert gn.shape[1] == ds.sizes["samples"]
    assert np.all(gn >= 0)
    assert np.all(gn <= 2)
    ac = ds["variant_allele_count"].values
    assert np.all(np.sum(gn, axis=1) == ac[:, 1])
    assert samples.ndim == 1
    assert samples.shape[0] == gn.shape[1]
    assert samples.tolist() == expected_samples

    return ds


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_sample_sets_param(
    fixture, api: AnophelesSnpData
):
    # Fixed parameters.
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

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
        check_biallelic_snp_calls_and_diplotypes(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_region_param(
    fixture, api: AnophelesSnpData
):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

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
        check_biallelic_snp_calls_and_diplotypes(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_site_mask_param(
    fixture, api: AnophelesSnpData
):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize site_mask.
    parametrize_site_mask = (None,) + api.site_mask_ids

    # Run tests.
    for site_mask in parametrize_site_mask:
        check_biallelic_snp_calls_and_diplotypes(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@pytest.mark.parametrize(
    "sample_query",
    ["sex_call == 'F'", "taxon == 'coluzzii'", "taxon == 'robot'"],
)
def test_biallelic_snp_calls_and_diplotypes_with_sample_query_param(
    ag3_sim_api: AnophelesSnpData, sample_query
):
    df_samples = ag3_sim_api.sample_metadata().query(sample_query)

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.biallelic_snp_calls(region="3L", sample_query=sample_query)

    else:
        ds = ag3_sim_api.biallelic_snp_calls(region="3L", sample_query=sample_query)
        assert ds.sizes["samples"] == len(df_samples)
        assert_array_equal(ds["sample_id"].values, df_samples["sample_id"].values)


@pytest.mark.parametrize(
    "sample_query,sample_query_options",
    [
        pytest.param(
            "sex_call in @sex_call_list", {"local_dict": {"sex_call_list": ["F", "M"]}}
        ),
        pytest.param(
            "taxon in @taxon_list",
            {"local_dict": {"taxon_list": ["coluzzii", "arabiensis"]}},
        ),
        pytest.param(
            "taxon in @non_taxon_list",
            {"local_dict": {"non_taxon_list": ["robot", "cyborg"]}},
        ),
    ],
)
def test_biallelic_snp_calls_and_diplotypes_with_sample_query_options_param(
    ag3_sim_api: AnophelesSnpData, sample_query, sample_query_options
):
    df_samples = ag3_sim_api.sample_metadata().query(
        sample_query, **sample_query_options
    )

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.biallelic_snp_calls(
                region="3L",
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )

    else:
        ds = ag3_sim_api.biallelic_snp_calls(
            region="3L",
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )
        assert ds.sizes["samples"] == len(df_samples)
        assert_array_equal(ds["sample_id"].values, df_samples["sample_id"].values)


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_min_cohort_size_param(
    fixture, api: AnophelesSnpData
):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with minimum cohort size.
    ds = api.biallelic_snp_calls(
        sample_sets=sample_sets,
        region=region,
        min_cohort_size=10,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["samples"] >= 10
    with pytest.raises(ValueError):
        api.biallelic_snp_calls(
            sample_sets=sample_sets,
            region=region,
            min_cohort_size=1_000,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_max_cohort_size_param(
    fixture, api: AnophelesSnpData
):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with maximum cohort size.
    ds = api.biallelic_snp_calls(
        sample_sets=sample_sets,
        region=region,
        max_cohort_size=15,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["samples"] <= 15


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_cohort_size_param(
    fixture, api: AnophelesSnpData
):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with specific cohort size.
    cohort_size = random.randint(1, 10)
    ds = api.biallelic_snp_calls(
        sample_sets=sample_sets,
        region=region,
        cohort_size=cohort_size,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.sizes["samples"] == cohort_size
    with pytest.raises(ValueError):
        api.biallelic_snp_calls(
            sample_sets=sample_sets,
            region=region,
            cohort_size=1_000,
        )


@pytest.mark.parametrize(
    "site_class",
    [
        "CDS_DEG_4",
        "CDS_DEG_2_SIMPLE",
        "CDS_DEG_0",
        "INTRON_SHORT",
        "INTRON_LONG",
        "INTRON_SPLICE_5PRIME",
        "INTRON_SPLICE_3PRIME",
        "UTR_5PRIME",
        "UTR_3PRIME",
        "INTERGENIC",
    ],
)
def test_biallelic_snp_calls_and_diplotypes_with_site_class_param(
    ag3_sim_api: AnophelesSnpData, site_class
):
    contig = random.choice(ag3_sim_api.contigs)
    ds1 = ag3_sim_api.biallelic_snp_calls(region=contig)
    ds2 = ag3_sim_api.biallelic_snp_calls(region=contig, site_class=site_class)
    assert ds2.sizes["variants"] < ds1.sizes["variants"]
    check_biallelic_snp_calls_and_diplotypes(
        ag3_sim_api, region=contig, site_class=site_class
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_conditions(
    fixture, api: AnophelesSnpData
):
    # Fixed parameters.
    contig = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrise conditions.
    min_minor_ac = random.randint(1, 3)
    max_missing_an = random.randint(5, 10)

    # Run tests.
    ds = check_biallelic_snp_calls_and_diplotypes(
        api=api,
        sample_sets=sample_sets,
        region=contig,
        site_mask=site_mask,
        min_minor_ac=min_minor_ac,
        max_missing_an=max_missing_an,
    )

    # Check conditions are met.
    ac = ds["variant_allele_count"].values
    ac_min = ac.min(axis=1)
    assert np.all(ac_min >= min_minor_ac)
    an = ac.sum(axis=1)
    an_missing = (ds.sizes["samples"] * ds.sizes["ploidy"]) - an
    assert np.all(an_missing <= max_missing_an)
    gt = ds["call_genotype"].values
    ac_check = allel.GenotypeArray(gt).count_alleles(max_allele=1)
    assert np.all(ac == ac_check)

    # Run tests with thinning.
    n_snps_available = int(ds.sizes["variants"])
    # This should always be true, although depends on min_minor_ac and max_missing_an,
    # so the range of values for those parameters needs to be chosen with some care.
    assert n_snps_available > 2
    n_snps_requested = random.randint(1, n_snps_available // 2)
    ds_thinned = check_biallelic_snp_calls_and_diplotypes(
        api=api,
        sample_sets=sample_sets,
        region=contig,
        site_mask=site_mask,
        min_minor_ac=min_minor_ac,
        max_missing_an=max_missing_an,
        n_snps=n_snps_requested,
    )
    n_snps_thinned = ds_thinned.sizes["variants"]
    assert n_snps_thinned >= n_snps_requested
    assert n_snps_thinned <= 2 * n_snps_requested

    # Ask for more SNPs than available.
    with pytest.raises(ValueError):
        api.biallelic_snp_calls(
            sample_sets=sample_sets,
            region=contig,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_snps=n_snps_available + 10,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_and_diplotypes_with_conditions_fractional(
    fixture, api: AnophelesSnpData
):
    # Fixed parameters.
    contig = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrise conditions.
    min_minor_ac = random.uniform(0, 0.05)
    max_missing_an = random.uniform(0.05, 0.2)

    # Run tests.
    ds = check_biallelic_snp_calls_and_diplotypes(
        api=api,
        sample_sets=sample_sets,
        region=contig,
        site_mask=site_mask,
        min_minor_ac=min_minor_ac,
        max_missing_an=max_missing_an,
    )

    # Check conditions are met.
    ac = ds["variant_allele_count"].values
    an = ac.sum(axis=1)
    ac_min = ac.min(axis=1)
    assert np.all((ac_min / an) >= min_minor_ac)
    an_missing = (ds.sizes["samples"] * ds.sizes["ploidy"]) - an
    assert np.all((an_missing / an) <= max_missing_an)
    gt = ds["call_genotype"].values
    ac_check = allel.GenotypeArray(gt).count_alleles(max_allele=1)
    assert np.all(ac == ac_check)

    # Run tests with thinning.
    n_snps_available = int(ds.sizes["variants"])
    # This should always be true, although depends on min_minor_ac and max_missing_an,
    # so the range of values for those parameters needs to be chosen with some care.
    assert n_snps_available > 2
    n_snps_requested = random.randint(1, n_snps_available // 2)
    ds_thinned = check_biallelic_snp_calls_and_diplotypes(
        api=api,
        sample_sets=sample_sets,
        region=contig,
        site_mask=site_mask,
        min_minor_ac=min_minor_ac,
        max_missing_an=max_missing_an,
        n_snps=n_snps_requested,
    )
    n_snps_thinned = ds_thinned.sizes["variants"]
    assert n_snps_thinned >= n_snps_requested
    assert n_snps_thinned <= 2 * n_snps_requested

    # Ask for more SNPs than available.
    with pytest.raises(ValueError):
        api.biallelic_snp_calls(
            sample_sets=sample_sets,
            region=contig,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_snps=n_snps_available + 10,
        )
