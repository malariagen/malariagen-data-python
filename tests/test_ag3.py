import random
import shutil

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from malariagen_data import Ag3, Region
from malariagen_data.ag3 import _cn_mode
from malariagen_data.util import locate_region, resolve_region

contigs = "2R", "2L", "3R", "3L", "X"


def setup_ag3(url="simplecache::gs://vo_agam_release/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Ag3 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Ag3(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Ag3(url, **kwargs)


def test_repr():
    ag3 = setup_ag3(check_location=True)
    assert isinstance(ag3, Ag3)
    r = repr(ag3)
    assert isinstance(r, str)


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_genome_sequence_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    seq_r = ag3.genome_sequence(region=contig_r)
    seq_l = ag3.genome_sequence(region=contig_l)
    seq = ag3.genome_sequence(region=chrom)
    assert isinstance(seq, da.Array)
    assert seq.dtype == "S1"
    assert seq.shape[0] == seq_r.shape[0] + seq_l.shape[0]
    # N.B., we use a single-threaded computation here to avoid race conditions
    # when data are being cached locally from GCS (which manifests as blosc
    # decompression errors).
    assert da.all(seq == da.concatenate([seq_r, seq_l])).compute(
        scheduler="single-threaded"
    )


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
def test_genome_sequence_joined_arms_region(region):
    ag3 = setup_ag3()
    seq = ag3.genome_sequence(region=region)
    assert isinstance(seq, da.Array)
    assert seq.dtype == "S1"
    assert seq.shape[0] == 1_000_001


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_genome_features_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    df_r = ag3.genome_features(region=contig_r)
    df_l = ag3.genome_features(region=contig_l)
    max_r = ag3.genome_sequence(contig_r).shape[0]
    df_l = df_l.assign(start=lambda x: x.start + max_r, end=lambda x: x.end + max_r)
    df_concat = pd.concat([df_r, df_l], axis=0).reset_index(drop=True)
    df_concat = df_concat.assign(contig=chrom)
    df = ag3.genome_features(region=chrom)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == df_r.shape[0] + df_l.shape[0]
    assert all(df["contig"] == chrom)
    assert_frame_equal(df, df_concat)


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
def test_genome_features_joined_arms_region(region):
    ag3 = setup_ag3()
    df = ag3.genome_features(region=region)
    assert isinstance(df, pd.DataFrame)
    assert df["contig"].unique() == region.split(":")[0]


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_sites_for_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    sites_r = ag3.snp_sites(region=contig_r, field="POS")
    sites_l = ag3.snp_sites(region=contig_l, field="POS")
    max_r = ag3.genome_sequence(region=contig_r).shape[0]
    sites_expected = da.concatenate([sites_r, sites_l + max_r])
    sites_actual = ag3.snp_sites(region=chrom, field="POS")

    assert isinstance(sites_actual, da.Array)
    assert sites_actual.ndim == 1
    assert sites_actual.dtype == "int32"
    assert da.all(sites_expected == sites_actual).compute(scheduler="single-threaded")


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
@pytest.mark.parametrize("field", ["POS", "REF", "ALT"])
def test_snp_sites_for_joined_arms_region(region, field):
    ag3 = setup_ag3()
    sites = ag3.snp_sites(region=region, field=field)

    start, end = region.replace(",", "").split(":")[1].split("-")
    size = int(end) - int(start)

    assert isinstance(sites, da.Array)
    assert sites.shape[0] <= size
    if field == "POS":
        assert sites.dtype == "int32"
        assert sites.ndim == 1
    elif field == "REF":
        assert sites.dtype == "S1"
        assert sites.ndim == 1
    elif field == "ALT":
        assert sites.dtype == "S1"
        assert sites.ndim == 2


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_genotypes_for_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    d_r = ag3.snp_genotypes(region=contig_r)
    d_l = ag3.snp_genotypes(region=contig_l)
    d = da.concatenate([d_r, d_l])

    gt = ag3.snp_genotypes(region=chrom)

    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert gt.shape == d.shape


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
def test_snp_genotypes_for_joined_arms_region(region):
    ag3 = setup_ag3()
    gt = ag3.snp_genotypes(region=region)
    sites = ag3.snp_sites(region=region, field="POS")

    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert sites.shape[0] == gt.shape[0]


def test_cross_metadata():
    ag3 = setup_ag3()
    df_crosses = ag3.cross_metadata()
    assert isinstance(df_crosses, pd.DataFrame)
    expected_cols = ["cross", "sample_id", "father_id", "mother_id", "sex", "role"]
    assert df_crosses.columns.tolist() == expected_cols

    # check samples are in AG1000G-X
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-X")
    assert set(df_crosses["sample_id"]) == set(df_samples["sample_id"])

    # check values
    expected_role_values = ["parent", "progeny"]
    assert df_crosses["role"].unique().tolist() == expected_role_values
    expected_sex_values = ["F", "M"]
    assert df_crosses["sex"].unique().tolist() == expected_sex_values


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_calls_for_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    ds_r = ag3.snp_calls(region=contig_r)
    ds_l = ag3.snp_calls(region=contig_l)
    ds_expected = xr.concat([ds_r, ds_l], dim="variants")

    ds_actual = ag3.snp_calls(region=chrom)

    assert isinstance(ds_actual, xr.Dataset)
    assert len(ds_actual.dims) == 4
    assert ds_actual["call_genotype"].dtype == "int8"
    assert ds_actual["variant_position"].dtype == "int32"
    assert ds_actual["call_genotype"].shape == ds_expected["call_genotype"].shape


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
def test_snp_calls_for_joined_arms_region(region):
    ag3 = setup_ag3()
    ds_snps = ag3.snp_calls(region=region)
    sites = ag3.snp_sites(region=region, field="POS")

    assert isinstance(ds_snps, xr.Dataset)
    assert len(ds_snps.dims) == 4
    assert ds_snps["call_genotype"].dtype == "int8"
    assert ds_snps["variant_position"].dtype == "int32"
    assert sites.shape[0] == ds_snps["call_genotype"].shape[0]


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_snp_variants_for_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    ds_r = ag3.snp_variants(region=contig_r)
    ds_l = ag3.snp_variants(region=contig_l)
    ds_expected = xr.concat([ds_r, ds_l], dim="variants")

    ds_actual = ag3.snp_variants(region=chrom)

    assert isinstance(ds_actual, xr.Dataset)
    assert len(ds_actual.dims) == 2
    assert ds_actual["variant_position"].dtype == "int32"
    assert ds_actual["variant_position"].shape == ds_expected["variant_position"].shape


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
def test_snp_variants_for_joined_arms_region(region):
    ag3 = setup_ag3()
    ds_vars = ag3.snp_variants(region=region)
    sites = ag3.snp_sites(region=region, field="POS")

    assert isinstance(ds_vars, xr.Dataset)
    assert len(ds_vars.dims) == 2
    assert ds_vars["variant_position"].dtype == "int32"
    assert sites.shape[0] == ds_vars["variant_position"].shape[0]


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_haplotypes_for_joined_arms(chrom):
    ag3 = setup_ag3()
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    ds_r = ag3.haplotypes(region=contig_r)
    ds_l = ag3.haplotypes(region=contig_l)
    ds_expected = xr.concat([ds_r, ds_l], dim="variants")

    ds_actual = ag3.haplotypes(region=chrom)

    assert isinstance(ds_actual, xr.Dataset)
    assert len(ds_actual.dims) == 4
    assert ds_actual["call_genotype"].dtype == "int8"
    assert ds_actual["variant_position"].dtype == "int32"
    assert ds_actual["call_genotype"].shape == ds_expected["call_genotype"].shape


@pytest.mark.parametrize(
    "region", ["2RL:61,000,000-62,000,000", "3RL:53,000,000-54,000,000"]
)
def test_haplotypes_for_joined_arms_region(region):
    ag3 = setup_ag3()
    ds_haps = ag3.haplotypes(region=region)

    assert isinstance(ds_haps, xr.Dataset)
    assert len(ds_haps.dims) == 4
    assert ds_haps["call_genotype"].dtype == "int8"
    assert ds_haps["variant_position"].dtype == "int32"


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
    utr_intron5 = "AGAP004679-RB"
    df = ag3.snp_effects(transcript=utr_intron5, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields
    assert df.shape == (7686, len(expected_fields))
    assert df.iloc[180].effect == "SPLICE_CORE"
    assert df.iloc[198].effect == "SPLICE_REGION"
    assert df.iloc[202].effect == "INTRONIC"

    # check 3' utr intron
    utr_intron3 = "AGAP000689-RA"
    df = ag3.snp_effects(transcript=utr_intron3, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields
    assert df.shape == (5397, len(expected_fields))
    assert df.iloc[646].effect == "SPLICE_CORE"
    assert df.iloc[652].effect == "SPLICE_REGION"
    assert df.iloc[674].effect == "INTRONIC"


def test_snp_allele_frequencies__dict_cohorts():
    ag3 = setup_ag3(cohorts_analysis="20230516")
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
    ag3 = setup_ag3(cohorts_analysis="20230516")
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
        min_cohort_size=min_cohort_size,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=True,
        effects=True,
    )
    df_coh = ag3.cohorts_metadata(sample_sets="3.0")
    coh_nm = "cohort_" + cohorts
    coh_counts = df_coh[coh_nm].dropna().value_counts()
    cohort_labels = coh_counts[coh_counts >= min_cohort_size].index.to_list()
    frq_cohort_labels = ["frq_" + s for s in cohort_labels]
    expected_fields = universal_fields + frq_cohort_labels + ["max_af"] + effects_fields

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 16641
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
        "aa_change",
    ]


def test_snp_allele_frequencies__query():
    ag3 = setup_ag3(cohorts_analysis="20230516")
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
        min_cohort_size=min_cohort_size,
        site_mask="gamb_colu",
        sample_sets="3.0",
        drop_invariant=True,
        effects=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert len(df) == 695


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
@pytest.mark.parametrize(
    "region", ["2R", "X", ["2R", "3R"], "3R:28,000,000-29,000,000"]
)
def test_gene_cnv(region, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(
        region=region, sample_sets=sample_sets, max_coverage_variance=None
    )

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
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    assert ds.dims["samples"] == n_samples
    df_genome_features = ag3.genome_features(region=region)
    df_genes = df_genome_features.query("type == 'gene'")
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
            assert x.ndim == 1
            assert x.dims == ("genes",)
        elif f.startswith("CN"):
            assert x.ndim == 2
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
@pytest.mark.parametrize("region", ["2R", "X", "3R:28,000,000-29,000,000"])
def test_gene_cnv_xarray_indexing(region, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(
        region=region, sample_sets=sample_sets, max_coverage_variance=None
    )

    # check label-based indexing
    # pick a random gene and sample ID

    # check dim lengths
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    df_genome_features = ag3.genome_features(region=region)
    df_genes = df_genome_features.query("type == 'gene'")
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


def _check_frequency(x):
    loc_nan = np.isnan(x)
    assert np.all(x[~loc_nan] >= 0)
    assert np.all(x[~loc_nan] <= 1)


@pytest.mark.parametrize(
    "region", ["2R", "X", ["2R", "3R"], "3R:28,000,000-29,000,000"]
)
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
def test_gene_cnv_frequencies(region, cohorts):
    ag3 = setup_ag3(cohorts_analysis="20230516")

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
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")

    df_cnv_frq = ag3.gene_cnv_frequencies(
        region=region,
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
        df_coh = ag3.cohorts_metadata(sample_sets="3.0")
        coh_nm = "cohort_" + cohorts
        frq_cols = ["frq_" + s for s in list(df_coh[coh_nm].dropna().unique())]

    # check frequencies are within sensible range
    for f in frq_cols:
        _check_frequency(df_cnv_frq[f].values)

    # check amp and del frequencies are within sensible range
    df_frq_amp = df_cnv_frq[frq_cols].xs("amp", level="cnv_type")
    df_frq_del = df_cnv_frq[frq_cols].xs("del", level="cnv_type")
    df_frq_sum = df_frq_amp + df_frq_del
    for f in frq_cols:
        _check_frequency(df_frq_sum[f].values)
    expected_fields = universal_fields + frq_cols
    assert sorted(df_cnv_frq.columns.tolist()) == sorted(expected_fields)


def test_gene_cnv_frequencies__query():
    ag3 = setup_ag3(cohorts_analysis="20230516")

    region = "3L"

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

    df = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets="3.0",
        cohorts="admin1_year",
        min_cohort_size=10,
        sample_query="country == 'Angola'",
        drop_invariant=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")
    assert len(df) == len(df_genes) * 2


def test_gene_cnv_frequencies__max_coverage_variance():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    region = "3L"
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")

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
        region=region,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
        cohorts="admin1_year",
        min_cohort_size=10,
        max_coverage_variance=None,
        drop_invariant=False,
    )
    expected_frq_columns = [
        "frq_GM-L_gcx2_2006",
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
        region=region,
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
    ag3 = setup_ag3(cohorts_analysis="20230516")
    region = "3L"

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

    df = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets="3.0",
        cohorts="admin1_year",
        min_cohort_size=10,
        sample_query="country == 'Angola'",
        drop_invariant=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert np.all(df["max_af"] > 0)
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")
    assert len(df) < len(df_genes) * 2


def test_gene_cnv_frequencies__dup_samples():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    df_dup = ag3.gene_cnv_frequencies(
        region="3L",
        cohorts="admin1_year",
        sample_sets=["AG1000G-FR", "AG1000G-FR"],
    )
    df = ag3.gene_cnv_frequencies(
        region="3L",
        cohorts="admin1_year",
        sample_sets=["AG1000G-FR"],
    )
    assert_frame_equal(df, df_dup)


def test_gene_cnv_frequencies__multi_contig_x():
    # https://github.com/malariagen/malariagen-data-python/issues/166

    ag3 = setup_ag3(cohorts_analysis="20230516")

    df1 = ag3.gene_cnv_frequencies(
        region="X",
        sample_sets="AG1000G-BF-B",
        cohorts="admin1_year",
        min_cohort_size=10,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    df2 = ag3.gene_cnv_frequencies(
        region=["2R", "X"],
        sample_sets="AG1000G-BF-B",
        cohorts="admin1_year",
        min_cohort_size=10,
        drop_invariant=False,
        max_coverage_variance=None,
    ).query("contig == 'X'")

    assert_frame_equal(df1, df2)


def test_gene_cnv_frequencies__missing_samples():
    # https://github.com/malariagen/malariagen-data-python/issues/183

    ag3 = setup_ag3(cohorts_analysis="20230516", pre=True)

    df = ag3.gene_cnv_frequencies(
        region="3L",
        sample_sets="1190-VO-GH-AMENGA-ETEGO-VMF00013",
        cohorts="admin1_year",
    )
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "region_raw",
    [
        "AGAP007280",
        "3L",
        "2R:48714463-48715355",
        "2R:24,630,355-24,633,221",
        Region("2R", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
    ag3 = setup_ag3()
    gene_annotation = ag3.genome_features(attributes=["ID"])
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

    # check that gene name matches coordinates from the genome_features and matches gene sequence
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
    if region_raw == "2R:24,630,355-24,633,221":
        assert region == Region("2R", 24630355, 24633221)


def test_aa_allele_frequencies():
    ag3 = setup_ag3(cohorts_analysis="20230516")

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
    ag3 = setup_ag3(cohorts_analysis="20230516")

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
    ag3 = setup_ag3(cohorts_analysis="20230516")

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
    region="2L",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=["AG1000G-BF-A", "AG1000G-ML-A", "AG1000G-UG"],
    sample_query=None,
    min_cohort_size=10,
    variant_query="max_af > 0.02",
    drop_invariant=True,
    max_coverage_variance=0.2,
):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    ds = ag3.gene_cnv_frequencies_advanced(
        region=region,
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
            region=region,
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


@pytest.mark.parametrize("region", ["2R", "X", ["3R", "X"], "3R:28,000,000-29,000,000"])
def test_gene_cnv_frequencies_advanced__region(region):
    _check_gene_cnv_frequencies_advanced(
        region=region,
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

    ag3 = setup_ag3(cohorts_analysis="20230516")

    ds1 = ag3.gene_cnv_frequencies_advanced(
        region="X",
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
        region=["2R", "X"],
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


def test_gene_cnv_frequencies_advanced__missing_samples():
    # https://github.com/malariagen/malariagen-data-python/issues/183

    ag3 = setup_ag3(cohorts_analysis="20230516", pre=True)

    ds = ag3.gene_cnv_frequencies_advanced(
        region="3L",
        sample_sets="1190-VO-GH-AMENGA-ETEGO-VMF00013",
        area_by="admin1_iso",
        period_by="year",
    )
    assert isinstance(ds, xr.Dataset)


def test_gene_cnv_frequencies_advanced__dup_samples():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    ds_dup = ag3.gene_cnv_frequencies_advanced(
        region="3L",
        area_by="admin1_iso",
        period_by="year",
        sample_sets=["AG1000G-BF-A", "AG1000G-BF-A"],
    )
    ds = ag3.gene_cnv_frequencies_advanced(
        region="3L",
        area_by="admin1_iso",
        period_by="year",
        sample_sets=["AG1000G-BF-A"],
    )
    assert ds.dims == ds_dup.dims


@pytest.mark.parametrize(
    "region",
    [
        "2R:1,000,000-2,000,000",
        "AGAP004707",
        ["2R:1,000,000-2,000,000", "2L:1,000,000-2,000,000"],
    ],
)
@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-AO", ["AG1000G-BF-A", "AG1000G-BF-B"]]
)
@pytest.mark.parametrize("sample_query", [None, "taxon == 'coluzzii'"])
@pytest.mark.parametrize("site_mask", [None, "gamb_colu_arab"])
def test_pca(region, sample_sets, sample_query, site_mask):
    results_cache = "../results_cache"
    shutil.rmtree(results_cache, ignore_errors=True)
    ag3 = setup_ag3(results_cache=results_cache)

    n_components = 8
    df_pca, evr = ag3.pca(
        region=region,
        n_snps=100,
        sample_sets=sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
        n_components=n_components,
    )

    df_samples = ag3.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
    )

    assert isinstance(df_pca, pd.DataFrame)
    assert len(df_pca) == len(df_samples)
    expected_columns = df_samples.columns.tolist() + [
        f"PC{n+1}" for n in range(n_components)
    ]
    assert df_pca.columns.tolist() == expected_columns
    assert_frame_equal(df_samples, df_pca[df_samples.columns.tolist()])
    assert isinstance(evr, np.ndarray)
    assert evr.shape == (n_components,)

    df_pca2, evr2 = ag3.pca(
        region=region,
        n_snps=100,
        sample_sets=sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
        n_components=n_components,
    )
    assert_frame_equal(df_pca, df_pca2)
    assert_array_equal(evr, evr2)


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


def test_h12_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    sample_query = "country == 'Ghana'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, h12 = ag3.h12_gwss(
        contig=contig,
        analysis=analysis,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        cohort_size=20,
    )

    # check dataset
    assert isinstance(x, np.ndarray)
    assert isinstance(h12, np.ndarray)

    # check dimensions
    assert len(x) == 11354
    assert len(x) == len(h12)

    # check some values
    assert_allclose(x[0], 27701.195)
    assert_allclose(h12[11353], 0.17875)


def test_h1x_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    cohort1_query = "cohort_admin2_year == 'ML-2_Kati_colu_2014'"
    cohort2_query = "cohort_admin2_year == 'ML-2_Kati_gamb_2014'"
    contig = "2L"
    analysis = "gamb_colu"
    window_size = 2000

    x, h1x = ag3.h1x_gwss(
        contig=contig,
        analysis=analysis,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        cohort_size=None,
    )

    # check data
    assert isinstance(x, np.ndarray)
    assert isinstance(h1x, np.ndarray)

    # check dimensions
    assert x.ndim == h1x.ndim == 1
    assert x.shape == h1x.shape

    # check some values
    assert_allclose(x[0], 36493.229, rtol=1e-5), x[0]
    assert_allclose(h1x[0], 0.067621, rtol=1e-5), h1x[0]
    assert np.all(h1x <= 1)
    assert np.all(h1x >= 0)


def test_average_fst():
    ag3 = setup_ag3()
    region = "3L"
    cohort1_query = "cohort_admin2_year == 'ML-2_Kati_colu_2014'"
    cohort2_query = "cohort_admin2_year == 'ML-2_Kati_gamb_2014'"
    n_jack = 200
    site_mask = "gamb_colu"

    fst_hudson, se_hudson = ag3.average_fst(
        region=region,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        n_jack=n_jack,
        site_mask=site_mask,
    )

    # check data
    assert isinstance(fst_hudson, np.float64)
    assert isinstance(se_hudson, np.float64)

    # check dimensions
    assert fst_hudson.ndim == se_hudson.ndim == 0

    # check some values
    assert_allclose(fst_hudson, 0.039983, rtol=1e5), fst_hudson
    assert_allclose(se_hudson, 0.003327, rtol=1e5), se_hudson
    assert np.all(fst_hudson <= 1)
    assert np.all(fst_hudson >= -0.05)


def test_pairwise_average_fst():
    ag3 = setup_ag3()
    region = "3L"
    cohorts = "cohort_admin1_year"
    sample_query = "country == 'Mali' and taxon == 'gambiae'"
    n_jack = 200
    site_mask = "gamb_colu"

    test_df = pd.DataFrame(
        {
            "cohort1_query": [
                "ML-2_gamb_2004",
                "ML-2_gamb_2014",
                "ML-2_gamb_2014",
                "ML-3_gamb_2012",
                "ML-3_gamb_2012",
                "ML-3_gamb_2012",
            ],
            "cohort2_query": [
                "ML-2_gamb_2004",
                "ML-2_gamb_2004",
                "ML-2_gamb_2014",
                "ML-2_gamb_2004",
                "ML-2_gamb_2014",
                "ML-3_gamb_2012",
            ],
            "Fst": [
                -0.05305554559810363,
                0.01033833066046641,
                -0.0530156630138291,
                0.010615727653386375,
                -0.00018451990702906661,
                -0.05300903322895467,
            ],
            "SE": [
                1.2398969283554441e-05,
                0.000964603651694087,
                1.1740037365959329e-05,
                0.0010044504116367143,
                0.00031950050471179716,
                1.2908421762356301e-05,
            ],
        }
    )

    fst_df = ag3.pairwise_average_fst(
        region=region,
        cohorts=cohorts,
        sample_query=sample_query,
        n_jack=n_jack,
        site_mask=site_mask,
    )

    # check data
    assert isinstance(fst_df, pd.core.frame.DataFrame)

    # check dimensions
    assert fst_df.ndim == 2
    assert fst_df["Fst"].shape == fst_df["SE"].shape

    # check some values
    pd.testing.assert_frame_equal(fst_df, test_df, rtol=1e5)
    assert np.all(fst_df["Fst"] <= 1)
    assert np.all(fst_df["Fst"] >= -0.05)


def test_fst_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    cohort1_query = "cohort_admin2_year == 'ML-2_Kati_colu_2014'"
    cohort2_query = "cohort_admin2_year == 'ML-2_Kati_gamb_2014'"
    contig = "2L"
    site_mask = "gamb_colu"
    window_size = 10_000

    x, fst = ag3.fst_gwss(
        contig=contig,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        site_mask=site_mask,
        cohort_size=None,
    )

    # check data
    assert isinstance(x, np.ndarray)
    assert isinstance(fst, np.ndarray)

    # check dimensions
    assert x.ndim == fst.ndim == 1
    assert x.shape == fst.shape

    # check some values
    assert_allclose(x[0], 56835.9649, rtol=1e-5), x[0]
    assert_allclose(fst[0], 0.0405522778148594, rtol=1e-5), fst[0]
    assert np.all(fst <= 1)
    assert np.all(np.logical_and(fst >= -0.1, fst <= 1))


def test_ihs_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    sample_query = "country == 'Ghana'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, ihs = ag3.ihs_gwss(
        contig=contig,
        analysis=analysis,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        max_cohort_size=20,
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)

    # check dimensions
    assert len(x) == 395
    assert len(x) == len(ihs)

    # check some values
    assert_allclose(x[0], 510232.847)
    assert_allclose(ihs[:, 2][100], 2.3467595962486327)


def test_xpehh_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    cohort1_query = "country == 'Ghana'"
    cohort2_query = "country == 'Angola'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, xpehh = ag3.xpehh_gwss(
        contig=contig,
        analysis=analysis,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        sample_sets=sample_sets,
        window_size=window_size,
        max_cohort_size=20,
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(xpehh, np.ndarray)

    # check dimensions
    assert len(x) == 399
    assert len(x) == len(xpehh)

    # check some values
    assert_allclose(x[0], 467448.348)
    assert_allclose(xpehh[:, 2][100], 0.4817561326426265)


def test_g123_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    sample_query = "country == 'Ghana'"
    contig = "3L"
    site_mask = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, g123 = ag3.g123_gwss(
        contig=contig,
        sites=site_mask,
        site_mask=site_mask,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        min_cohort_size=20,
        max_cohort_size=50,
    )

    # check dataset
    assert isinstance(x, np.ndarray)
    assert isinstance(g123, np.ndarray)

    # check dimensions
    assert len(x) == 11354
    assert len(x) == len(g123)

    # check some values
    assert_allclose(x[0], 27701.195)
    assert_allclose(g123[11353], 0.18799999999999997)
