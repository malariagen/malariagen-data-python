import random

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr
import zarr
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from malariagen_data import Ag3, Region
from malariagen_data.ag3 import _cn_mode
from malariagen_data.util import locate_region, resolve_region

expected_species = {
    "gambiae",
    "coluzzii",
    "arabiensis",
    "intermediate_arabiensis_gambiae",
    "intermediate_gambiae_coluzzii",
}


contigs = "2R", "2L", "3R", "3L", "X"


def setup_ag3(url="simplecache::gs://vo_agam_release/", **kwargs):
    if url is None:
        # test default URL
        return Ag3(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Ag3(url, **kwargs)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "gs://vo_agam_release/",
        "gcs://vo_agam_release/",
        "gs://vo_agam_release",
        "gcs://vo_agam_release",
        "simplecache::gs://vo_agam_release/",
        "simplecache::gcs://vo_agam_release/",
    ],
)
def test_sample_sets(url):

    ag3 = setup_ag3(url)
    df_sample_sets_v3 = ag3.sample_sets(release="3.0")
    assert isinstance(df_sample_sets_v3, pd.DataFrame)
    assert len(df_sample_sets_v3) == 28
    assert tuple(df_sample_sets_v3.columns) == ("sample_set", "sample_count", "release")

    # test duplicates not allowed
    with pytest.raises(ValueError):
        ag3.sample_sets(release=["3.0", "3.0"])

    # test default is all public releases
    df_default = ag3.sample_sets()
    df_all = ag3.sample_sets(release=ag3.releases)
    assert_frame_equal(df_default, df_all)


def test_releases():

    ag3 = setup_ag3()
    assert isinstance(ag3.releases, tuple)
    assert ag3.releases == ("3.0",)

    ag3 = setup_ag3(pre=True)
    assert isinstance(ag3.releases, tuple)
    assert len(ag3.releases) > 1
    assert all([r.startswith("3.") for r in ag3.releases])


def test_sample_metadata():

    ag3 = setup_ag3()
    df_sample_sets_v3 = ag3.sample_sets(release="3.0")

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

    # all v3
    df_samples_v3 = ag3.sample_metadata(
        sample_sets="3.0", species_analysis=None, cohorts_analysis=None
    )
    assert tuple(df_samples_v3.columns) == expected_cols
    expected_len = df_sample_sets_v3["sample_count"].sum()
    assert len(df_samples_v3) == expected_len

    # single sample set
    df_samples_x = ag3.sample_metadata(
        sample_sets="AG1000G-X", species_analysis=None, cohorts_analysis=None
    )
    assert tuple(df_samples_x.columns) == expected_cols
    expected_len = df_sample_sets_v3.query("sample_set == 'AG1000G-X'")[
        "sample_count"
    ].sum()
    assert len(df_samples_x) == expected_len

    # multiple sample sets
    sample_sets = ["AG1000G-BF-A", "AG1000G-BF-B", "AG1000G-BF-C"]
    df_samples_bf = ag3.sample_metadata(
        sample_sets=sample_sets, species_analysis=None, cohorts_analysis=None
    )
    assert tuple(df_samples_bf) == expected_cols
    loc_sample_sets = df_sample_sets_v3["sample_set"].isin(sample_sets)
    expected_len = df_sample_sets_v3.loc[loc_sample_sets]["sample_count"].sum()
    assert len(df_samples_bf) == expected_len

    # duplicate sample sets
    with pytest.raises(ValueError):
        ag3.sample_metadata(
            sample_sets=["3.0", "3.0"], species_analysis=None, cohorts_analysis=None
        )
    with pytest.raises(ValueError):
        ag3.sample_metadata(
            sample_sets=["AG1000G-UG", "AG1000G-UG"],
            species_analysis=None,
            cohorts_analysis=None,
        )
    with pytest.raises(ValueError):
        ag3.sample_metadata(
            sample_sets=["AG1000G-UG", "3.0"],
            species_analysis=None,
            cohorts_analysis=None,
        )

    # default is all public releases
    df_default = ag3.sample_metadata(species_analysis=None, cohorts_analysis=None)
    df_all = ag3.sample_metadata(
        sample_sets=ag3.releases, species_analysis=None, cohorts_analysis=None
    )
    assert_frame_equal(df_default, df_all)

    aim_cols = (
        "aim_species_fraction_colu",
        "aim_species_fraction_arab",
        "aim_species_gambcolu_arabiensis",
        "aim_species_gambiae_coluzzii",
        "aim_species",
    )

    # AIM species calls, included by default
    df_samples_aim = ag3.sample_metadata(sample_sets="3.0", cohorts_analysis=None)
    assert tuple(df_samples_aim.columns) == expected_cols + aim_cols
    assert len(df_samples_aim) == len(df_samples_v3)
    assert set(df_samples_aim["aim_species"].dropna()) == expected_species

    # AIM species calls, explicit
    df_samples_aim = ag3.sample_metadata(
        sample_sets="3.0", species_analysis="aim_20200422", cohorts_analysis=None
    )
    assert tuple(df_samples_aim.columns) == expected_cols + aim_cols
    assert len(df_samples_aim) == len(df_samples_v3)
    assert set(df_samples_aim["aim_species"].dropna()) == expected_species

    pca_cols = (
        "pca_species_pc1",
        "pca_species_pc2",
        "pca_species_gambcolu_arabiensis",
        "pca_species_gambiae_coluzzii",
        "pca_species",
    )

    # PCA species calls
    df_samples_pca = ag3.sample_metadata(
        sample_sets="3.0", species_analysis="pca_20200422", cohorts_analysis=None
    )
    assert tuple(df_samples_pca.columns) == expected_cols + pca_cols
    assert len(df_samples_pca) == len(df_samples_v3)
    assert (
        set(df_samples_pca["pca_species"].dropna()).difference(expected_species)
        == set()
    )

    cohort_cols = (
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
    # cohort calls
    df_samples_coh = ag3.sample_metadata(
        sample_sets="3.0", species_analysis=None, cohorts_analysis="20211101"
    )
    assert tuple(df_samples_coh.columns) == expected_cols + cohort_cols
    assert len(df_samples_coh) == len(df_samples_v3)


@pytest.mark.parametrize(
    "sample_sets",
    [
        "AG1000G-AO",
        "AG1000G-X",
        ["AG1000G-BF-A", "AG1000G-BF-B"],
        "3.0",
        None,
    ],
)
@pytest.mark.parametrize("analysis", ["aim_20200422", "pca_20200422"])
def test_species_calls(sample_sets, analysis):
    ag3 = setup_ag3()
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    df_species = ag3.species_calls(sample_sets=sample_sets, analysis=analysis)
    assert len(df_species) == len(df_samples)
    if analysis.startswith("aim_"):
        assert (
            set(df_species["aim_species"].dropna()).difference(expected_species)
            == set()
        )
    if analysis.startswith("pca_"):
        assert (
            set(df_species["pca_species"].dropna()).difference(expected_species)
            == set()
        )


@pytest.mark.parametrize("mask", ["gamb_colu_arab", "gamb_colu", "arab"])
def test_open_site_filters(mask):
    # check can open the zarr directly
    ag3 = setup_ag3()
    root = ag3.open_site_filters(mask=mask)
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in ag3.contigs:
        assert contig in root


@pytest.mark.parametrize("mask", ["gamb_colu_arab", "gamb_colu", "arab"])
@pytest.mark.parametrize(
    "region", ["2R", ["3R", "3L", "2R:48,714,463-48,715,355", "AGAP007280"]]
)
def test_site_filters(mask, region):
    ag3 = setup_ag3()
    filter_pass = ag3.site_filters(region=region, mask=mask)
    assert isinstance(filter_pass, da.Array)
    assert filter_pass.ndim == 1
    assert filter_pass.dtype == bool


def test_open_snp_sites():
    ag3 = setup_ag3()
    root = ag3.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in ag3.contigs:
        assert contig in root


@pytest.mark.parametrize("chunks", ["auto", "native"])
@pytest.mark.parametrize(
    "region", ["2R", ["3R", "2R:48,714,463-48,715,355", "AGAP007280"]]
)
def test_snp_sites(chunks, region):

    ag3 = setup_ag3()

    pos = ag3.snp_sites(region=region, field="POS", chunks=chunks)
    ref = ag3.snp_sites(region=region, field="REF", chunks=chunks)
    alt = ag3.snp_sites(region=region, field="ALT", chunks=chunks)
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

    # apply site mask
    filter_pass = ag3.site_filters(region=region, mask="gamb_colu_arab").compute()
    n_pass = np.count_nonzero(filter_pass)
    pos_pass = ag3.snp_sites(
        region=region, field="POS", site_mask="gamb_colu_arab", chunks=chunks
    )
    assert isinstance(pos_pass, da.Array)
    assert pos_pass.ndim == 1
    assert pos_pass.dtype == "i4"
    assert pos_pass.shape[0] == n_pass
    assert pos_pass.compute().shape == pos_pass.shape
    for f in "POS", "REF", "ALT":
        d = ag3.snp_sites(
            region=region, site_mask="gamb_colu_arab", field=f, chunks=chunks
        )
        assert isinstance(d, da.Array)
        assert d.shape[0] == n_pass
        assert d.shape == d.compute().shape


def test_open_snp_genotypes():
    # check can open the zarr directly
    ag3 = setup_ag3()
    root = ag3.open_snp_genotypes(sample_set="AG1000G-AO")
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in ag3.contigs:
        assert contig in root


@pytest.mark.parametrize("chunks", ["auto", "native"])
@pytest.mark.parametrize(
    "sample_sets",
    [None, "AG1000G-X", ["AG1000G-BF-A", "AG1000G-BF-B"], "3.0"],
)
@pytest.mark.parametrize(
    "region", ["2R", ["3R", "2R:48,714,463-48,715,355", "AGAP007280"]]
)
def test_snp_genotypes(chunks, sample_sets, region):

    ag3 = setup_ag3()

    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    gt = ag3.snp_genotypes(region=region, sample_sets=sample_sets, chunks=chunks)
    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert gt.shape[1] == len(df_samples)

    # specific fields
    x = ag3.snp_genotypes(
        region=region, sample_sets=sample_sets, field="GT", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i1"
    x = ag3.snp_genotypes(
        region=region, sample_sets=sample_sets, field="GQ", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "i2"
    x = ag3.snp_genotypes(
        region=region, sample_sets=sample_sets, field="MQ", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "i2"
    x = ag3.snp_genotypes(
        region=region, sample_sets=sample_sets, field="AD", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i2"

    # site mask
    filter_pass = ag3.site_filters(region=region, mask="gamb_colu_arab").compute()
    gt_pass = ag3.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        site_mask="gamb_colu_arab",
        chunks=chunks,
    )
    assert isinstance(gt_pass, da.Array)
    assert gt_pass.ndim == 3
    assert gt_pass.dtype == "i1"
    assert gt_pass.shape[0] == np.count_nonzero(filter_pass)
    assert gt_pass.shape[1] == len(df_samples)
    assert gt_pass.shape[2] == 2


@pytest.mark.parametrize(
    "sample_sets",
    [None, "AG1000G-X", ["AG1000G-BF-A", "AG1000G-BF-B"], "3.0"],
)
@pytest.mark.parametrize(
    "region", ["2R", ["3R", "2R:48,714,463-48,715,355", "AGAP007280"]]
)
def test_snp_genotypes_chunks(sample_sets, region):

    ag3 = setup_ag3()
    gt_native = ag3.snp_genotypes(
        region=region, sample_sets=sample_sets, chunks="native"
    )
    gt_auto = ag3.snp_genotypes(region=region, sample_sets=sample_sets, chunks="auto")
    gt_manual = ag3.snp_genotypes(
        region=region, sample_sets=sample_sets, chunks=(100_000, 10, 2)
    )

    assert gt_native.chunks != gt_auto.chunks
    assert gt_auto.chunks != gt_manual.chunks
    assert gt_manual.chunks != gt_native.chunks
    assert gt_manual.chunks[0][0] == 100_000
    assert gt_manual.chunks[1][0] == 10
    assert gt_manual.chunks[2][0] == 2


def test_genome():

    ag3 = setup_ag3()

    # test the open_genome() method to access as zarr
    genome = ag3.open_genome()
    assert isinstance(genome, zarr.hierarchy.Group)
    for contig in ag3.contigs:
        assert contig in genome
        assert genome[contig].dtype == "S1"

    # test the genome_sequence() method to access sequences
    for contig in ag3.contigs:
        seq = ag3.genome_sequence(contig)
        assert isinstance(seq, da.Array)
        assert seq.dtype == "S1"


def test_geneset():

    ag3 = setup_ag3()

    # default
    df = ag3.geneset()
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
    expected_cols = gff3_cols + ["ID", "Parent", "Name", "description"]
    assert df.columns.tolist() == expected_cols

    # don't unpack attributes
    df = ag3.geneset(attributes=None)
    assert isinstance(df, pd.DataFrame)
    expected_cols = gff3_cols + ["attributes"]
    assert df.columns.tolist() == expected_cols


@pytest.mark.parametrize(
    "region",
    ["AGAP007280", "2R:48714463-48715355", "2R", "X"],
)
@pytest.mark.parametrize("mask", ["gamb_colu_arab", "gamb_colu", "arab"])
def test_is_accessible(region, mask):

    ag3 = setup_ag3()
    # run a couple of tests
    is_accessible = ag3.is_accessible(region=region, site_mask=mask)
    assert isinstance(is_accessible, np.ndarray)
    assert is_accessible.ndim == 1
    assert is_accessible.shape[0] == ag3.genome_sequence(region).shape[0]


def test_cross_metadata():

    ag3 = setup_ag3()
    df_crosses = ag3.cross_metadata()
    assert isinstance(df_crosses, pd.DataFrame)
    expected_cols = ["cross", "sample_id", "father_id", "mother_id", "sex", "role"]
    assert df_crosses.columns.tolist() == expected_cols

    # check samples are in AG1000G-X
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-X", species_analysis=None)
    assert set(df_crosses["sample_id"]) == set(df_samples["sample_id"])

    # check values
    expected_role_values = ["parent", "progeny"]
    assert df_crosses["role"].unique().tolist() == expected_role_values
    expected_sex_values = ["F", "M"]
    assert df_crosses["sex"].unique().tolist() == expected_sex_values


def test_site_annotations():

    ag3 = setup_ag3()

    # test access as zarr
    root = ag3.open_site_annotations()
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
        for contig in contigs:
            assert contig in root[f]

    # test access as dask arrays
    for region in "2R", "X", "AGAP007280", "2R:48714463-48715355":
        for site_mask in None, "gamb_colu_arab":
            pos = ag3.snp_sites(region=region, field="POS", site_mask=site_mask)
            for field in "codon_degeneracy", "seq_cls":
                d = ag3.site_annotations(
                    region=region, field=field, site_mask=site_mask
                )
                assert isinstance(d, da.Array)
                assert d.ndim == 1
                assert d.shape == pos.shape


@pytest.mark.parametrize(
    "sample_sets",
    [None, "AG1000G-X", ["AG1000G-BF-A", "AG1000G-BF-B"], "3.0"],
)
@pytest.mark.parametrize(
    "region", ["2R", ["3R", "2R:48,714,463-48,715,355", "AGAP007280"]]
)
@pytest.mark.parametrize("site_mask", [None, "gamb_colu_arab"])
def test_snp_calls(sample_sets, region, site_mask):

    ag3 = setup_ag3()

    ds = ag3.snp_calls(region=region, sample_sets=sample_sets, site_mask=site_mask)
    assert isinstance(ds, xr.Dataset)

    # check fields
    expected_data_vars = {
        "variant_allele",
        "variant_filter_pass_gamb_colu_arab",
        "variant_filter_pass_gamb_colu",
        "variant_filter_pass_arab",
        "call_genotype",
        "call_genotype_mask",
        "call_GQ",
        "call_AD",
        "call_MQ",
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
    pos = ag3.snp_sites(region=region, field="POS", site_mask=site_mask)
    n_variants = len(pos)
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    n_samples = len(df_samples)
    assert ds.dims["variants"] == n_variants
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 4

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert x.ndim, f == 2
            assert x.shape == (n_variants, 4)
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim, f == 1
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
            assert x.ndim, f == 2
            assert x.dims == ("variants", "samples")
            assert x.shape == (n_variants, n_samples)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # check samples
    expected_samples = df_samples["sample_id"].tolist()
    assert ds["sample_id"].values.tolist() == expected_samples

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xr.DataArray)

    # check compress bug
    pos = ds["variant_position"].data
    assert pos.shape == pos.compute().shape


def test_snp_effects():
    ag3 = setup_ag3()
    gste2 = "AGAP009194-RA"
    site_mask = "gamb_colu"
    expected_fields = [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
        "transcript",
        "effect",
        "impact",
        "ref_codon",
        "alt_codon",
        "aa_pos",
        "ref_aa",
        "alt_aa",
        "aa_change",
    ]

    df = ag3.snp_effects(transcript=gste2, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields

    # reverse strand gene
    assert df.shape == (2838, len(expected_fields))
    # check first, second, third codon position non-syn
    assert df.iloc[1454].aa_change == "I114L"
    assert df.iloc[1446].aa_change == "I114M"
    # while we are here, check all columns for a position
    assert df.iloc[1451].position == 28598166
    assert df.iloc[1451].ref_allele == "A"
    assert df.iloc[1451].alt_allele == "G"
    assert df.iloc[1451].effect == "NON_SYNONYMOUS_CODING"
    assert df.iloc[1451].impact == "MODERATE"
    assert df.iloc[1451].ref_codon == "aTt"
    assert df.iloc[1451].alt_codon == "aCt"
    assert df.iloc[1451].aa_pos == 114
    assert df.iloc[1451].ref_aa == "I"
    assert df.iloc[1451].alt_aa == "T"
    assert df.iloc[1451].aa_change == "I114T"
    # check syn
    assert df.iloc[1447].aa_change == "I114I"
    # check intronic
    assert df.iloc[1197].effect == "INTRONIC"
    # check 5' utr
    assert df.iloc[2661].effect == "FIVE_PRIME_UTR"
    # check 3' utr
    assert df.iloc[0].effect == "THREE_PRIME_UTR"

    # test forward strand gene gste6
    gste6 = "AGAP009196-RA"
    df = ag3.snp_effects(transcript=gste6, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields
    assert df.shape == (2829, len(expected_fields))

    # check first, second, third codon position non-syn
    assert df.iloc[701].aa_change == "E35*"
    assert df.iloc[703].aa_change == "E35V"
    # while we are here, check all columns for a position
    assert df.iloc[706].position == 28600605
    assert df.iloc[706].ref_allele == "G"
    assert df.iloc[706].alt_allele == "C"
    assert df.iloc[706].effect == "NON_SYNONYMOUS_CODING"
    assert df.iloc[706].impact == "MODERATE"
    assert df.iloc[706].ref_codon == "gaG"
    assert df.iloc[706].alt_codon == "gaC"
    assert df.iloc[706].aa_pos == 35
    assert df.iloc[706].ref_aa == "E"
    assert df.iloc[706].alt_aa == "D"
    assert df.iloc[706].aa_change == "E35D"
    # check syn
    assert df.iloc[705].aa_change == "E35E"
    # check intronic
    assert df.iloc[900].effect == "INTRONIC"
    # check 5' utr
    assert df.iloc[0].effect == "FIVE_PRIME_UTR"
    # check 3' utr
    assert df.iloc[2828].effect == "THREE_PRIME_UTR"

    # check 5' utr intron and the different intron effects
    utrintron5 = "AGAP004679-RB"
    df = ag3.snp_effects(transcript=utrintron5, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields
    assert df.shape == (7686, len(expected_fields))
    assert df.iloc[180].effect == "SPLICE_CORE"
    assert df.iloc[198].effect == "SPLICE_REGION"
    assert df.iloc[202].effect == "INTRONIC"

    # check 3' utr intron
    utrintron3 = "AGAP000689-RA"
    df = ag3.snp_effects(transcript=utrintron3, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields
    assert df.shape == (5397, len(expected_fields))
    assert df.iloc[646].effect == "SPLICE_CORE"
    assert df.iloc[652].effect == "SPLICE_REGION"
    assert df.iloc[674].effect == "INTRONIC"


def test_snp_allele_frequencies__str_cohorts():
    ag3 = setup_ag3()
    cohorts = "admin1_month"
    min_cohort_size = 10
    universal_fields = [
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
        "label",
    ]
    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts=cohorts,
        cohorts_analysis="20211101",
        min_cohort_size=min_cohort_size,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=True,
        effects=False,
    )
    df_coh = ag3.sample_cohorts(sample_sets="3.0", cohorts_analysis="20211101")
    coh_nm = "cohort_" + cohorts
    coh_counts = df_coh[coh_nm].dropna().value_counts().to_frame()
    cohort_labels = coh_counts[coh_counts[coh_nm] >= min_cohort_size].index.to_list()
    frq_cohort_labels = ["frq_" + s for s in cohort_labels]
    expected_fields = universal_fields + frq_cohort_labels + ["max_af"]

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == ["contig", "position", "ref_allele", "alt_allele"]
    assert len(df) == 16526


def test_snp_allele_frequencies__dict_cohorts():
    ag3 = setup_ag3()
    cohorts = {
        "ke": "country == 'Kenya'",
        "bf_2012_col": "country == 'Burkina Faso' and year == 2012 and aim_species == 'coluzzii'",
    }
    universal_fields = [
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
        "label",
    ]

    # test drop invariants
    df = ag3.snp_allele_frequencies(
        transcript="AGAP009194-RA",
        cohorts=cohorts,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=True,
        effects=False,
    )

    assert isinstance(df, pd.DataFrame)
    frq_columns = ["frq_" + s for s in list(cohorts.keys())]
    expected_fields = universal_fields + frq_columns + ["max_af"]
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.shape == (133, len(expected_fields))
    assert df.iloc[3].frq_ke == 0
    assert df.iloc[4].frq_bf_2012_col == pytest.approx(0.006097, abs=1e-6)
    assert df.iloc[4].max_af == pytest.approx(0.006097, abs=1e-6)
    # check invariant have been dropped
    assert df.max_af.min() > 0

    # test keep invariants
    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts=cohorts,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=False,
        effects=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.shape == (132306, len(expected_fields))
    # check invariant positions are still present
    assert np.any(df.max_af == 0)


def test_snp_allele_frequencies__str_cohorts__effects():
    ag3 = setup_ag3()
    cohorts = "admin1_month"
    min_cohort_size = 10
    universal_fields = [
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
        "label",
    ]
    effects_fields = [
        "transcript",
        "effect",
        "impact",
        "ref_codon",
        "alt_codon",
        "aa_pos",
        "ref_aa",
        "alt_aa",
    ]
    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts=cohorts,
        cohorts_analysis="20211101",
        min_cohort_size=min_cohort_size,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=True,
        effects=True,
    )
    df_coh = ag3.sample_cohorts(sample_sets="3.0", cohorts_analysis="20211101")
    coh_nm = "cohort_" + cohorts
    coh_counts = df_coh[coh_nm].dropna().value_counts().to_frame()
    cohort_labels = coh_counts[coh_counts[coh_nm] >= min_cohort_size].index.to_list()
    frq_cohort_labels = ["frq_" + s for s in cohort_labels]
    expected_fields = universal_fields + frq_cohort_labels + ["max_af"] + effects_fields

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 16526
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
        "aa_change",
    ]


def test_snp_allele_frequencies__query():
    ag3 = setup_ag3()
    cohorts = "admin1_year"
    min_cohort_size = 10
    expected_columns = [
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
        "frq_AO-LUA_colu_2009",
        "max_af",
        "label",
    ]

    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts=cohorts,
        sample_query="country == 'Angola'",
        cohorts_analysis="20211101",
        min_cohort_size=min_cohort_size,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=True,
        effects=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert len(df) == 695


def test_snp_allele_frequencies__dup_samples():
    ag3 = setup_ag3()
    with pytest.raises(ValueError):
        ag3.snp_allele_frequencies(
            transcript="AGAP004707-RD",
            cohorts="admin1_year",
            sample_sets=["AG1000G-FR", "AG1000G-FR"],
        )


@pytest.mark.parametrize(
    "sample_sets",
    ["AG1000G-AO", ["AG1000G-AO", "AG1000G-UG"], "3.0", None],
)
@pytest.mark.parametrize("region", ["2R", ["3L", "X"], "3R:28,000,000-29,000,000"])
def test_cnv_hmm(sample_sets, region):
    ag3 = setup_ag3()
    ds = ag3.cnv_hmm(region=region, sample_sets=sample_sets)
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
    if region in ag3.contigs:
        n_variants_expected = 1 + len(ag3.genome_sequence(region=region)) // 300
    elif isinstance(region, (tuple, list)) and all([r in ag3.contigs for r in region]):
        n_variants_expected = sum(
            [1 + len(ag3.genome_sequence(region=c)) // 300 for c in region]
        )
    else:
        # test part of a contig region
        region = ag3.resolve_region(region)
        variant_contig = ds["variant_contig"].values
        contig_index = ds.attrs["contigs"].index(region.contig)
        assert np.all(variant_contig == contig_index)
        variant_position = ds["variant_position"].values
        variant_end = ds["variant_end"].values
        assert variant_position[0] <= region.start
        assert variant_end[0] >= region.start
        assert variant_position[-1] <= region.end
        assert variant_end[-1] >= region.end
        assert np.all(variant_position <= region.end)
        assert np.all(variant_end >= region.start)
        n_variants_expected = 1 + (region.end - region.start) // 300

    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    n_samples_expected = len(df_samples)
    assert ds.dims["variants"] == n_variants_expected
    assert ds.dims["samples"] == n_samples_expected

    # check sample IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f.startswith("variant_"):
            assert x.ndim, f == 1
            assert x.shape == (n_variants_expected,)
            assert x.dims == ("variants",)
        elif f.startswith("call_"):
            assert x.ndim, f == 2
            assert x.dims == ("variants", "samples")
            assert x.shape == (n_variants_expected, n_samples_expected)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples_expected,)

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can set up computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_CN"].sum(axis=1)
    assert isinstance(d2, xr.DataArray)


@pytest.mark.parametrize("sample_set", ["AG1000G-AO", "AG1000G-UG", "AG1000G-X"])
@pytest.mark.parametrize("analysis", ["gamb_colu", "arab", "crosses"])
@pytest.mark.parametrize("contig", ["3L", "X", ["2R", "2L"]])
def test_cnv_coverage_calls(sample_set, analysis, contig):

    ag3 = setup_ag3()

    expected_analyses = {
        "AG1000G-AO": {"gamb_colu"},
        "AG1000G-UG": {"gamb_colu", "arab"},
        "AG1000G-X": {"crosses"},
    }
    if analysis not in expected_analyses[sample_set]:
        with pytest.raises(ValueError):
            ag3.cnv_coverage_calls(
                contig=contig, analysis=analysis, sample_set=sample_set
            )
        return

    ds = ag3.cnv_coverage_calls(contig=contig, analysis=analysis, sample_set=sample_set)
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
    df_samples = ag3.sample_metadata(sample_sets=sample_set, species_analysis=None)
    sample_id = pd.Series(ds["sample_id"].values)
    assert sample_id.isin(df_samples["sample_id"]).all()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f.startswith("variant_"):
            assert x.ndim, f == 1
            assert x.dims == ("variants",)
        elif f.startswith("call_"):
            assert x.ndim, f == 2
            assert x.dims == ("variants", "samples")
        elif f.startswith("sample_"):
            assert x.ndim, f == 1
            assert x.dims == ("samples",)

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can set up computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_genotype"].sum(axis=1)
    assert isinstance(d2, xr.DataArray)


@pytest.mark.parametrize(
    "sample_sets",
    [
        "AG1000G-AO",
        "AG1000G-UG",
        ["AG1000G-AO", "AG1000G-UG"],
        "3.0",
        None,
    ],
)
@pytest.mark.parametrize("contig", ["2R", "3R", "X", ["2R", "3R"]])
def test_cnv_discordant_read_calls(sample_sets, contig):

    ag3 = setup_ag3()

    if contig == "3L":
        with pytest.raises(ValueError):
            ag3.cnv_discordant_read_calls(contig=contig, sample_sets=sample_sets)
        return

    ds = ag3.cnv_discordant_read_calls(contig=contig, sample_sets=sample_sets)
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
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    n_samples = len(df_samples)
    assert ds.dims["samples"] == n_samples

    expected_variants = {"2R": 40, "3R": 29, "X": 29}
    if isinstance(contig, str):
        n_variants = expected_variants[contig]
    elif isinstance(contig, (list, tuple)):
        n_variants = sum([expected_variants[c] for c in contig])
    else:
        raise NotImplementedError

    assert ds.dims["variants"] == n_variants

    # check sample IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f.startswith("variant_"):
            assert x.ndim, f == 1
            assert x.dims == ("variants",)
        elif f.startswith("call_"):
            assert x.ndim, f == 2
            assert x.dims == ("variants", "samples")
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can set up computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_genotype"].sum(axis=1)
    assert isinstance(d2, xr.DataArray)


@pytest.mark.parametrize(
    "sample_sets",
    ["AG1000G-AO", ["AG1000G-AO", "AG1000G-UG"], "3.0", None],
)
@pytest.mark.parametrize("contig", ["2L", "3L"])
def test_cnv_discordant_read_calls__no_calls(sample_sets, contig):

    ag3 = setup_ag3()

    with pytest.raises(ValueError):
        ag3.cnv_discordant_read_calls(contig=contig, sample_sets=sample_sets)
    return


@pytest.mark.parametrize("rows", [10, 100, 1000])
@pytest.mark.parametrize("cols", [10, 100, 1000])
@pytest.mark.parametrize("vmax", [2, 12, 100])
def test_cn_mode(rows, cols, vmax):
    """Test the numba-optimised function for computing modal copy number."""

    a = np.random.randint(0, vmax, size=(rows * cols), dtype="i1").reshape(rows, cols)
    expect = scipy.stats.mode(a, axis=0)
    modes, counts = _cn_mode(a, vmax)
    assert_array_equal(expect.mode.squeeze(), modes)
    assert_array_equal(expect.count.squeeze(), counts)


# noinspection PyArgumentList
@pytest.mark.parametrize(
    "sample_sets",
    ["AG1000G-AO", ("AG1000G-TZ", "AG1000G-UG"), "3.0", None],
)
@pytest.mark.parametrize("contig", ["2R", "X", ["2R", "3R"]])
def test_gene_cnv(contig, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(contig=contig, sample_sets=sample_sets)

    assert isinstance(ds, xr.Dataset)

    # check fields
    expected_data_vars = {
        "CN_mode",
        "CN_mode_count",
        "gene_windows",
        "gene_contig",
        "gene_start",
        "gene_end",
        "gene_name",
        "gene_description",
        "gene_strand",
        "sample_coverage_variance",
        "sample_is_high_variance",
    }
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "gene_id",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # check dimensions
    assert set(ds.dims) == {"samples", "genes"}

    # check dim lengths
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    n_samples = len(df_samples)
    assert ds.dims["samples"] == n_samples
    df_geneset = ag3.geneset()
    if isinstance(contig, str):
        df_genes = df_geneset.query(f"type == 'gene' and contig == '{contig}'")
    else:
        df_genes = pd.concat(
            [df_geneset.query(f"type == 'gene' and contig == '{c}'") for c in contig]
        )
    n_genes = len(df_genes)
    assert ds.dims["genes"] == n_genes

    # check IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()
    assert ds["gene_id"].values.tolist() == df_genes["ID"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, np.ndarray)

        if f.startswith("gene_"):
            assert x.ndim, f == 1
            assert x.dims == ("genes",)
        elif f.startswith("CN"):
            assert x.ndim, f == 2
            assert x.dims == ("genes", "samples")
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # check can set up computations
    d1 = ds["gene_start"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["CN_mode"].max(axis=1)
    assert isinstance(d2, xr.DataArray)

    # sanity checks
    x = ds["gene_windows"].values
    y = ds["CN_mode_count"].values.max(axis=1)
    assert np.all(x >= y)
    z = ds["CN_mode"].values
    assert np.max(z) <= 12
    assert np.min(z) >= -1


@pytest.mark.parametrize(
    "sample_sets",
    ["AG1000G-AO", ("AG1000G-TZ", "AG1000G-UG"), "3.0", None],
)
@pytest.mark.parametrize("contig", ["2R", "X"])
def test_gene_cnv_xarray_indexing(contig, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(contig=contig, sample_sets=sample_sets)

    # check label-based indexing
    # pick a random gene and sample ID

    # check dim lengths
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_analysis=None)
    df_geneset = ag3.geneset()
    df_genes = df_geneset.query(f"type == 'gene' and contig == '{contig}'")
    gene = random.choice(df_genes["ID"].tolist())
    sample = random.choice(df_samples["sample_id"].tolist())
    ds = ds.set_index(genes="gene_id", samples="sample_id")
    o = ds.sel(genes=gene)
    assert isinstance(o, xr.Dataset)
    assert set(o.dims) == {"samples"}
    assert o.dims["samples"] == ds.dims["samples"]
    o = ds.sel(samples=sample)
    assert isinstance(o, xr.Dataset)
    assert set(o.dims) == {"genes"}
    assert o.dims["genes"] == ds.dims["genes"]
    o = ds.sel(genes=gene, samples=sample)
    assert isinstance(o, xr.Dataset)
    assert set(o.dims) == set()


@pytest.mark.parametrize("contig", ["2R", "X", ["2R", "3R"]])
@pytest.mark.parametrize(
    "cohorts",
    [
        {
            "ke": "country == 'Kenya'",
            "bf_2012_col": "country == 'Burkina Faso' and year == 2012 and aim_species == 'coluzzii'",
        },
        "admin1_month",
    ],
)
def test_gene_cnv_frequencies(contig, cohorts):

    universal_fields = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
    ]
    ag3 = setup_ag3()
    if isinstance(contig, str):
        df_genes = ag3.geneset().query(f"type == 'gene' and contig == '{contig}'")
    else:
        df_genes = pd.concat(
            [ag3.geneset().query(f"type == 'gene' and contig == '{c}'") for c in contig]
        )

    df_cnv_frq = ag3.gene_cnv_frequencies(
        contig=contig,
        sample_sets="3.0",
        cohorts=cohorts,
        min_cohort_size=1,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    assert isinstance(df_cnv_frq, pd.DataFrame)
    assert len(df_cnv_frq) == len(df_genes) * 2
    assert df_cnv_frq.index.names == ["gene_id", "gene_name", "cnv_type"]

    # sanity checks
    frq_cols = None
    if isinstance(cohorts, dict):
        frq_cols = ["frq_" + s for s in cohorts.keys()]
    if isinstance(cohorts, str):
        df_coh = ag3.sample_cohorts(sample_sets="3.0", cohorts_analysis="20211101")
        coh_nm = "cohort_" + cohorts
        frq_cols = ["frq_" + s for s in list(df_coh[coh_nm].dropna().unique())]

    # check frequencies are within sensible range
    for f in frq_cols:
        x = df_cnv_frq[f].values
        assert np.all(x >= 0), f
        assert np.all(x <= 1), f

    # check amp and del frequencies are within sensible range
    df_frq_amp = df_cnv_frq[frq_cols].xs("amp", level="cnv_type")
    df_frq_del = df_cnv_frq[frq_cols].xs("del", level="cnv_type")
    df_frq_sum = df_frq_amp + df_frq_del
    for f in frq_cols:
        x = df_frq_sum[f].values
        assert np.all(x >= 0)
        assert np.all(x <= 1)
    expected_fields = universal_fields + frq_cols
    assert sorted(df_cnv_frq.columns.tolist()) == sorted(expected_fields)


def test_gene_cnv_frequencies__query():

    contig = "3L"

    expected_columns = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
        "frq_AO-LUA_colu_2009",
    ]

    ag3 = setup_ag3()
    df = ag3.gene_cnv_frequencies(
        contig=contig,
        sample_sets="3.0",
        cohorts="admin1_year",
        min_cohort_size=10,
        sample_query="country == 'Angola'",
        drop_invariant=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    df_genes = ag3.geneset().query(f"type == 'gene' and contig == '{contig}'")
    assert len(df) == len(df_genes) * 2


def test_gene_cnv_frequencies__max_coverage_variance():

    ag3 = setup_ag3()
    contig = "3L"
    df_genes = ag3.geneset().query(f"type == 'gene' and contig == '{contig}'")

    base_columns = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
    ]

    # run without a threshold on coverage variance
    df = ag3.gene_cnv_frequencies(
        contig=contig,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
        cohorts="admin1_year",
        min_cohort_size=10,
        max_coverage_variance=None,
        drop_invariant=False,
    )
    expected_frq_columns = [
        "frq_GM-L_gcx2_2012",
        "frq_GM-M_gcx2_2012",
        "frq_GM-N_gcx1_2011",
    ]
    expected_columns = base_columns + expected_frq_columns
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert len(df) == len(df_genes) * 2

    # Run with a threshold on coverage variance - this will remove samples,
    # which in turn will drop one of the cohorts below the min_cohort_size,
    # and so we can check that we have lost a cohort.
    df = ag3.gene_cnv_frequencies(
        contig=contig,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
        cohorts="admin1_year",
        min_cohort_size=10,
        max_coverage_variance=0.2,
        drop_invariant=False,
    )
    expected_frq_columns = [
        "frq_GM-M_gcx2_2012",
        "frq_GM-N_gcx1_2011",
    ]
    expected_columns = base_columns + expected_frq_columns
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert len(df) == len(df_genes) * 2


def test_gene_cnv_frequencies__drop_invariant():

    contig = "3L"

    expected_columns = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
        "frq_AO-LUA_colu_2009",
    ]

    ag3 = setup_ag3()
    df = ag3.gene_cnv_frequencies(
        contig=contig,
        sample_sets="3.0",
        cohorts="admin1_year",
        min_cohort_size=10,
        sample_query="country == 'Angola'",
        drop_invariant=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert np.all(df["max_af"] > 0)
    df_genes = ag3.geneset().query(f"type == 'gene' and contig == '{contig}'")
    assert len(df) < len(df_genes) * 2


def test_gene_cnv_frequencies__dup_samples():
    ag3 = setup_ag3()
    with pytest.raises(ValueError):
        ag3.gene_cnv_frequencies(
            contig="3L",
            cohorts="admin1_year",
            sample_sets=["AG1000G-FR", "AG1000G-FR"],
        )


def test_gene_cnv_frequencies__multi_contig_x():
    # https://github.com/malariagen/malariagen-data-python/issues/166

    ag3 = setup_ag3()

    df1 = ag3.gene_cnv_frequencies(
        contig="X",
        sample_sets="AG1000G-BF-B",
        cohorts="admin1_year",
        min_cohort_size=10,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    df2 = ag3.gene_cnv_frequencies(
        contig=["2R", "X"],
        sample_sets="AG1000G-BF-B",
        cohorts="admin1_year",
        min_cohort_size=10,
        drop_invariant=False,
        max_coverage_variance=None,
    ).query("contig == 'X'")

    assert_frame_equal(df1, df2)


@pytest.mark.parametrize(
    "sample_sets",
    ["AG1000G-BF-A", ("AG1000G-TZ", "AG1000G-UG"), "3.0", None],
)
@pytest.mark.parametrize(
    "region", ["2R", ["3R", "2R:48,714,463-48,715,355", "AGAP007280"]]
)
@pytest.mark.parametrize("analysis", ["arab", "gamb_colu", "gamb_colu_arab"])
def test_haplotypes(sample_sets, region, analysis):

    ag3 = setup_ag3()

    # check expected samples
    sample_query = None
    if analysis == "arab":
        sample_query = "aim_species == 'arabiensis' and sample_set != 'AG1000G-X'"
    elif analysis == "gamb_colu":
        sample_query = (
            "aim_species in ['gambiae', 'coluzzii', 'intermediate_gambiae_coluzzii'] and "
            "sample_set != 'AG1000G-X'"
        )
    elif analysis == "gamb_colu_arab":
        sample_query = "sample_set != 'AG1000G-X'"
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    expected_samples = df_samples.query(sample_query)["sample_id"].tolist()
    n_samples = len(expected_samples)

    # check if any samples
    if n_samples == 0:
        ds = ag3.haplotypes(region=region, sample_sets=sample_sets, analysis=analysis)
        assert ds is None
        return

    ds = ag3.haplotypes(region=region, sample_sets=sample_sets, analysis=analysis)
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

    # check samples
    samples = ds["sample_id"].values
    assert set(samples) == set(expected_samples)

    # check dim lengths
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 2

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert x.ndim, f == 2
            assert x.shape[1] == 2
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim, f == 1
            assert x.dims == ("variants",)
        elif f == "call_genotype":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape[1] == n_samples
            assert x.shape[2] == 2

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_genotype"].sum(axis=(1, 2))
    assert isinstance(d2, xr.DataArray)


# test v3 sample sets
@pytest.mark.parametrize(
    "sample_sets",
    ["3.0", "AG1000G-UG", ["AG1000G-AO", "AG1000G-FR"]],
)
def test_sample_cohorts(sample_sets):
    expected_cols = (
        "sample_id",
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

    ag3 = setup_ag3()
    df_coh = ag3.sample_cohorts(sample_sets=sample_sets, cohorts_analysis="20211101")
    df_meta = ag3.sample_metadata(sample_sets=sample_sets)

    assert tuple(df_coh.columns) == expected_cols
    assert len(df_coh) == len(df_meta)
    assert df_coh.sample_id.tolist() == df_meta.sample_id.tolist()
    if sample_sets == "AG1000G-UG":
        assert df_coh.sample_id[0] == "AC0007-C"
        assert df_coh.cohort_admin1_year[23] == "UG-E_arab_2012"
        assert df_coh.cohort_admin1_month[37] == "UG-E_arab_2012_10"
        assert df_coh.cohort_admin2_year[42] == "UG-E_Tororo_arab_2012"
        assert df_coh.cohort_admin2_month[49] == "UG-E_Tororo_arab_2012_10"
    if sample_sets == ["AG1000G-AO", "AG1000G-FR"]:
        assert df_coh.sample_id[0] == "AR0047-C"
        assert df_coh.sample_id[103] == "AP0017-Cx"


@pytest.mark.parametrize(
    "region_raw",
    [
        "AGAP007280",
        "3L",
        "2R:48714463-48715355",
        "2L:24,630,355-24,633,221",
        Region("2R", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):

    ag3 = setup_ag3()
    gene_annotation = ag3.geneset(["ID"])
    region = resolve_region(ag3, region_raw)
    pos = ag3.snp_sites(region=region.contig, field="POS")
    ref = ag3.snp_sites(region=region.contig, field="REF")
    loc_region = locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "3L":
        assert region.contig == "3L"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "AGAP007280":
        gene = gene_annotation.query("ID == 'AGAP007280'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute()
            == np.array(["A", "T", "G", "G", "C"], dtype="S1")
        ).all()

    # check string parsing
    if region_raw == "2R:48714463-48715355":
        assert region == Region("2R", 48714463, 48715355)
    if region_raw == "2L:24,630,355-24,633,221":
        assert region == Region("2L", 24630355, 24633221)


def test_aa_allele_frequencies():
    ag3 = setup_ag3()

    expected_fields = [
        "transcript",
        "aa_pos",
        "ref_allele",
        "alt_allele",
        "ref_aa",
        "alt_aa",
        "effect",
        "impact",
        "frq_BF-09_gamb_2012",
        "frq_BF-09_colu_2012",
        "frq_BF-09_colu_2014",
        "frq_BF-09_gamb_2014",
        "frq_BF-07_gamb_2004",
        "max_af",
        "label",
    ]

    df = ag3.aa_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts="admin1_year",
        cohorts_analysis="20211101",
        min_cohort_size=10,
        site_mask="gamb_colu",
        sample_sets=("AG1000G-BF-A", "AG1000G-BF-B", "AG1000G-BF-C"),
        drop_invariant=True,
    )

    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ["aa_change", "contig", "position"]
    assert df.shape == (61, len(expected_fields))
    assert df.loc["V402L"].max_af[0] == pytest.approx(0.121951, abs=1e-6)


def test_aa_allele_frequencies__dup_samples():
    ag3 = setup_ag3()
    with pytest.raises(ValueError):
        ag3.aa_allele_frequencies(
            transcript="AGAP004707-RD",
            cohorts="admin1_year",
            sample_sets=["AG1000G-FR", "AG1000G-FR"],
        )


# noinspection PyDefaultArgument
def _check_snp_allele_frequencies_advanced(
    transcript="AGAP004707-RD",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=["AG1000G-BF-A", "AG1000G-ML-A", "AG1000G-UG"],
    sample_query=None,
    min_cohort_size=10,
    nobs_mode="called",
    variant_query="max_af > 0.02",
):

    ag3 = setup_ag3()

    ds = ag3.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        min_cohort_size=min_cohort_size,
        nobs_mode=nobs_mode,
        variant_query=variant_query,
    )

    assert isinstance(ds, xr.Dataset)

    # noinspection PyTypeChecker
    assert sorted(ds.dims) == ["cohorts", "variants"]

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_position",
        "variant_ref_allele",
        "variant_alt_allele",
        "variant_max_af",
        "variant_pass_gamb_colu_arab",
        "variant_pass_gamb_colu",
        "variant_pass_arab",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_ref_codon",
        "variant_alt_codon",
        "variant_ref_aa",
        "variant_alt_aa",
        "variant_aa_pos",
        "variant_aa_change",
    )
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    expected_cohort_vars = (
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    )
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    expected_event_vars = (
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    )
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # sanity checks for area values
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    if sample_query is not None:
        df_samples = df_samples.query(sample_query)
    expected_area = np.unique(df_samples[area_by].dropna().values)
    area = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area:
        assert a in expected_area

    # sanity checks for period values
    period = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "A-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # sanity check cohort size
    size = ds["cohort_size"].values
    for s in size:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year" and nobs_mode == "called":

        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler snp_allele_frequencies()
        # function with the admin1_year cohorts.

        # check consistency with the basic snp allele frequencies method
        df_af = ag3.snp_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
        )
        df_af = df_af.reset_index()  # make sure all variables available to check
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # check cohorts are consistent
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # check variants are consistent
        assert ds.dims["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            _compare_series_like(actual, expect)

        # check frequencies are consistent
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


# noinspection PyDefaultArgument
def _check_aa_allele_frequencies_advanced(
    transcript="AGAP004707-RD",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=["AG1000G-BF-A", "AG1000G-ML-A", "AG1000G-UG"],
    sample_query=None,
    min_cohort_size=10,
    nobs_mode="called",
    variant_query="max_af > 0.02",
):

    ag3 = setup_ag3()

    ds = ag3.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        min_cohort_size=min_cohort_size,
        nobs_mode=nobs_mode,
        variant_query=variant_query,
    )

    assert isinstance(ds, xr.Dataset)

    # noinspection PyTypeChecker
    assert sorted(ds.dims) == ["cohorts", "variants"]

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_position",
        "variant_max_af",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_ref_aa",
        "variant_alt_aa",
        "variant_aa_pos",
        "variant_aa_change",
    )
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    expected_cohort_vars = (
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    )
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    expected_event_vars = (
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    )
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # sanity checks for area values
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    if sample_query is not None:
        df_samples = df_samples.query(sample_query)
    expected_area = np.unique(df_samples[area_by].dropna().values)
    area = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area:
        assert a in expected_area

    # sanity checks for period values
    period = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "A-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # sanity check cohort size
    size = ds["cohort_size"].values
    for s in size:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year" and nobs_mode == "called":

        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler aa_allele_frequencies()
        # function with the admin1_year cohorts.

        # check consistency with the basic snp allele frequencies method
        df_af = ag3.aa_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
        )
        df_af = df_af.reset_index()  # make sure all variables available to check
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # check cohorts are consistent
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # check variants are consistent
        assert ds.dims["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            _compare_series_like(actual, expect)

        # check frequencies are consistent
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            print(cohort_label)
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


# Here we don't explore the full matrix, but vary one parameter at a time, otherwise
# the test suite would take too long to run.


@pytest.mark.parametrize("transcript", ["AGAP004707-RD", "AGAP006028-RA"])
def test_allele_frequencies_advanced__transcript(transcript):
    _check_snp_allele_frequencies_advanced(
        transcript=transcript,
    )
    _check_aa_allele_frequencies_advanced(
        transcript=transcript,
    )


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
def test_allele_frequencies_advanced__area_by(area_by):
    _check_snp_allele_frequencies_advanced(
        area_by=area_by,
    )
    _check_aa_allele_frequencies_advanced(
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
def test_allele_frequencies_advanced__period_by(period_by):
    _check_snp_allele_frequencies_advanced(
        period_by=period_by,
    )
    _check_aa_allele_frequencies_advanced(
        period_by=period_by,
    )


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-BF-A", ["AG1000G-BF-A", "AG1000G-ML-A"], "3.0"]
)
def test_allele_frequencies_advanced__sample_sets(sample_sets):
    _check_snp_allele_frequencies_advanced(
        sample_sets=sample_sets,
    )
    _check_aa_allele_frequencies_advanced(
        sample_sets=sample_sets,
    )


@pytest.mark.parametrize(
    "sample_query",
    [
        "taxon in ['gambiae', 'coluzzii'] and country == 'Mali'",
        "taxon == 'arabiensis' and country in ['Uganda', 'Tanzania']",
    ],
)
def test_allele_frequencies_advanced__sample_query(sample_query):
    _check_snp_allele_frequencies_advanced(
        sample_query=sample_query,
    )
    # noinspection PyTypeChecker
    _check_aa_allele_frequencies_advanced(
        sample_query=sample_query,
        variant_query=None,
    )


@pytest.mark.parametrize("min_cohort_size", [10, 100])
def test_allele_frequencies_advanced__min_cohort_size(min_cohort_size):
    _check_snp_allele_frequencies_advanced(
        min_cohort_size=min_cohort_size,
    )
    _check_aa_allele_frequencies_advanced(
        min_cohort_size=min_cohort_size,
    )


@pytest.mark.parametrize(
    "variant_query",
    [
        None,
        "effect == 'NON_SYNONYMOUS_CODING' and max_af > 0.05",
        "effect == 'foobar'",  # no variants
    ],
)
def test_allele_frequencies_advanced__variant_query(variant_query):
    _check_snp_allele_frequencies_advanced(
        variant_query=variant_query,
    )
    _check_aa_allele_frequencies_advanced(
        variant_query=variant_query,
    )


@pytest.mark.parametrize("nobs_mode", ["called", "fixed"])
def test_allele_frequencies_advanced__nobs_mode(nobs_mode):
    _check_snp_allele_frequencies_advanced(
        nobs_mode=nobs_mode,
    )
    _check_aa_allele_frequencies_advanced(
        nobs_mode=nobs_mode,
    )


# noinspection PyDefaultArgument
def _check_gene_cnv_frequencies_advanced(
    contig="2L",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=["AG1000G-BF-A", "AG1000G-ML-A", "AG1000G-UG"],
    sample_query=None,
    min_cohort_size=10,
    variant_query="max_af > 0.02",
    drop_invariant=True,
    max_coverage_variance=0.2,
):

    ag3 = setup_ag3()

    ds = ag3.gene_cnv_frequencies_advanced(
        contig=contig,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        min_cohort_size=min_cohort_size,
        variant_query=variant_query,
        drop_invariant=drop_invariant,
        max_coverage_variance=max_coverage_variance,
    )

    assert isinstance(ds, xr.Dataset)

    # noinspection PyTypeChecker
    assert sorted(ds.dims) == ["cohorts", "variants"]

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_start",
        "variant_end",
        "variant_windows",
        "variant_cnv_type",
        "variant_gene_id",
        "variant_gene_name",
        "variant_gene_strand",
        "variant_max_af",
    )
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    expected_cohort_vars = (
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    )
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    expected_event_vars = (
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    )
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # sanity checks for area values
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    if sample_query is not None:
        df_samples = df_samples.query(sample_query)
    expected_area = np.unique(df_samples[area_by].dropna().values)
    area = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area:
        assert a in expected_area

    # sanity checks for period values
    period = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "A-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # sanity check cohort size
    size = ds["cohort_size"].values
    for s in size:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year":

        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler gene_cnv_frequencies()
        # function with the admin1_year cohorts.

        # check consistency with the basic gene CNV frequencies method
        df_af = ag3.gene_cnv_frequencies(
            contig=contig,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            drop_invariant=drop_invariant,
            max_coverage_variance=max_coverage_variance,
        )
        df_af = df_af.reset_index()  # make sure all variables available to check
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # check cohorts are consistent
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # check variants are consistent
        assert ds.dims["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            _compare_series_like(actual, expect)

        # check frequencies are consistent
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


@pytest.mark.parametrize("contig", ["2R", "X", ["3R", "X"]])
def test_gene_cnv_frequencies_advanced__contig(contig):
    _check_gene_cnv_frequencies_advanced(
        contig=contig,
    )


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
def test_gene_cnv_frequencies_advanced__area_by(area_by):
    _check_gene_cnv_frequencies_advanced(
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
def test_gene_cnv_frequencies_advanced__period_by(period_by):
    _check_gene_cnv_frequencies_advanced(
        period_by=period_by,
    )


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-BF-A", ["AG1000G-BF-A", "AG1000G-ML-A"], "3.0"]
)
def test_gene_cnv_frequencies_advanced__sample_sets(sample_sets):
    _check_gene_cnv_frequencies_advanced(
        sample_sets=sample_sets,
    )


@pytest.mark.parametrize(
    "sample_query",
    [
        "taxon in ['gambiae', 'coluzzii'] and country == 'Mali'",
        "taxon == 'arabiensis' and country in ['Uganda', 'Tanzania']",
    ],
)
def test_gene_cnv_frequencies_advanced__sample_query(sample_query):
    _check_gene_cnv_frequencies_advanced(
        sample_query=sample_query,
    )


@pytest.mark.parametrize("min_cohort_size", [10, 100])
def test_gene_cnv_frequencies_advanced__min_cohort_size(min_cohort_size):
    _check_gene_cnv_frequencies_advanced(
        min_cohort_size=min_cohort_size,
    )


@pytest.mark.parametrize(
    "variant_query",
    [
        None,
        "cnv_type == 'amp' and max_af > 0.05",
    ],
)
def test_gene_cnv_frequencies_advanced__variant_query(variant_query):
    _check_gene_cnv_frequencies_advanced(
        variant_query=variant_query,
    )


@pytest.mark.parametrize(
    "drop_invariant",
    [
        False,
        True,
    ],
)
def test_gene_cnv_frequencies_advanced__drop_invariant(drop_invariant):
    # noinspection PyTypeChecker
    _check_gene_cnv_frequencies_advanced(
        variant_query=None,
        drop_invariant=drop_invariant,
    )


@pytest.mark.parametrize(
    "max_coverage_variance",
    [None, 0.2],
)
def test_gene_cnv_frequencies_advanced__max_coverage_variance(max_coverage_variance):
    _check_gene_cnv_frequencies_advanced(
        max_coverage_variance=max_coverage_variance,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
    )


def test_gene_cnv_frequencies_advanced__multi_contig_x():
    # https://github.com/malariagen/malariagen-data-python/issues/166

    ag3 = setup_ag3()

    ds1 = ag3.gene_cnv_frequencies_advanced(
        contig="X",
        area_by="admin1_iso",
        period_by="year",
        sample_sets="AG1000G-BF-B",
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    ds2 = ag3.gene_cnv_frequencies_advanced(
        contig=["2R", "X"],
        area_by="admin1_iso",
        period_by="year",
        sample_sets="AG1000G-BF-B",
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=False,
        max_coverage_variance=None,
    )
    loc_x = ds2["variant_contig"].values == "X"
    ds2 = ds2.isel(variants=loc_x)

    for v in ds1:
        a = ds1[v]
        b = ds2[v]
        _compare_series_like(a, b)


def test_snp_allele_frequencies_advanced__dup_samples():
    ag3 = setup_ag3()
    with pytest.raises(ValueError):
        ag3.snp_allele_frequencies_advanced(
            transcript="AGAP004707-RD",
            area_by="admin1_iso",
            period_by="year",
            sample_sets=["AG1000G-BF-A", "AG1000G-BF-A"],
        )


def test_aa_allele_frequencies_advanced__dup_samples():
    ag3 = setup_ag3()
    with pytest.raises(ValueError):
        ag3.aa_allele_frequencies_advanced(
            transcript="AGAP004707-RD",
            area_by="admin1_iso",
            period_by="year",
            sample_sets=["AG1000G-BF-A", "AG1000G-BF-A"],
        )


def test_gene_cnv_frequencies_advanced__dup_samples():
    ag3 = setup_ag3()
    with pytest.raises(ValueError):
        ag3.gene_cnv_frequencies_advanced(
            contig="3L",
            area_by="admin1_iso",
            period_by="year",
            sample_sets=["AG1000G-BF-A", "AG1000G-BF-A"],
        )


# TODO test plot_frequencies...() functions


def _compare_series_like(actual, expect):

    # compare pandas series-like objects for equality or floating point
    # similarity, handling missing values appropriately

    # handle object arrays, these don't get nans compared properly
    t = actual.dtype
    if t == object:
        expect = expect.fillna("NA")
        actual = actual.fillna("NA")

    if t.kind == "f":
        assert_allclose(actual.values, expect.values)
    else:
        assert_array_equal(actual.values, expect.values)
