import random

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import scipy.stats
import xarray
import zarr
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from malariagen_data import Ag3
from malariagen_data.ag3 import _cn_mode

expected_species = {
    "gambiae",
    "coluzzii",
    "arabiensis",
    "intermediate_arabiensis_gambiae",
    "intermediate_gambiae_coluzzii",
}


contigs = "2R", "2L", "3R", "3L", "X"


def setup_ag3(url="simplecache::gs://vo_agam_release/", **storage_kwargs):
    if url.startswith("simplecache::"):
        storage_kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Ag3(url, **storage_kwargs)


@pytest.mark.parametrize(
    "url",
    [
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
    df_sample_sets_v3 = ag3.sample_sets(release="v3")
    assert isinstance(df_sample_sets_v3, pd.DataFrame)
    assert len(df_sample_sets_v3) == 28
    assert tuple(df_sample_sets_v3.columns) == ("sample_set", "sample_count", "release")

    # test default is v3
    df_default = ag3.sample_sets()
    assert_frame_equal(df_sample_sets_v3, df_default)


def test_sample_metadata():

    ag3 = setup_ag3()
    df_sample_sets_v3 = ag3.sample_sets(release="v3")

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
    df_samples_v3 = ag3.sample_metadata(sample_sets="v3", species_calls=None)
    assert tuple(df_samples_v3.columns) == expected_cols
    expected_len = df_sample_sets_v3["sample_count"].sum()
    assert len(df_samples_v3) == expected_len

    # v3_wild
    df_samples_v3_wild = ag3.sample_metadata(sample_sets="v3_wild", species_calls=None)
    assert tuple(df_samples_v3_wild.columns) == expected_cols
    expected_len = df_sample_sets_v3.query("sample_set != 'AG1000G-X'")[
        "sample_count"
    ].sum()
    assert len(df_samples_v3_wild) == expected_len

    # single sample set
    df_samples_x = ag3.sample_metadata(sample_sets="AG1000G-X", species_calls=None)
    assert tuple(df_samples_x.columns) == expected_cols
    expected_len = df_sample_sets_v3.query("sample_set == 'AG1000G-X'")[
        "sample_count"
    ].sum()
    assert len(df_samples_x) == expected_len

    # multiple sample sets
    sample_sets = ["AG1000G-BF-A", "AG1000G-BF-B", "AG1000G-BF-C"]
    df_samples_bf = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    assert tuple(df_samples_bf) == expected_cols
    loc_sample_sets = df_sample_sets_v3["sample_set"].isin(sample_sets)
    expected_len = df_sample_sets_v3.loc[loc_sample_sets]["sample_count"].sum()
    assert len(df_samples_bf) == expected_len

    # default is v3_wild
    df_default = ag3.sample_metadata(species_calls=None)
    assert_frame_equal(df_samples_v3_wild, df_default)

    aim_cols = (
        "aim_fraction_colu",
        "aim_fraction_arab",
        "species_gambcolu_arabiensis",
        "species_gambiae_coluzzii",
        "species",
    )

    # AIM species calls, included by default
    df_samples_aim = ag3.sample_metadata()
    assert tuple(df_samples_aim.columns) == expected_cols + aim_cols
    assert len(df_samples_aim) == len(df_samples_v3_wild)
    assert set(df_samples_aim["species"]) == expected_species

    # AIM species calls, explicit
    df_samples_aim = ag3.sample_metadata(species_calls=("20200422", "aim"))
    assert tuple(df_samples_aim.columns) == expected_cols + aim_cols
    assert len(df_samples_aim) == len(df_samples_v3_wild)
    assert set(df_samples_aim["species"]) == expected_species

    pca_cols = (
        "PC1",
        "PC2",
        "species_gambcolu_arabiensis",
        "species_gambiae_coluzzii",
        "species",
    )

    # PCA species calls
    df_samples_pca = ag3.sample_metadata(species_calls=("20200422", "pca"))
    assert tuple(df_samples_pca.columns) == expected_cols + pca_cols
    assert len(df_samples_pca) == len(df_samples_v3_wild)
    assert set(df_samples_pca["species"]).difference(expected_species) == set()


def test_species_calls():

    ag3 = setup_ag3()
    sample_sets = ag3.sample_sets(release="v3")["sample_set"].tolist()

    for s in sample_sets:
        for method in "aim", "pca":
            df_samples = ag3.sample_metadata(sample_sets=s, species_calls=None)
            df_species = ag3.species_calls(sample_sets=s, method=method)
            assert len(df_species) == len(df_samples)
            if s == "AG1000G-X":
                # no species calls
                assert df_species["species"].isna().all()
            else:
                assert not df_species["species"].isna().any()
                assert set(df_species["species"]).difference(expected_species) == set()


def test_site_filters():

    ag3 = setup_ag3()

    for mask in "gamb_colu_arab", "gamb_colu", "arab":

        # check can open the zarr directly
        root = ag3.open_site_filters(mask=mask)
        assert isinstance(root, zarr.hierarchy.Group)
        for contig in contigs:
            assert contig in root

        # check access as dask array
        for contig in contigs:
            filter_pass = ag3.site_filters(contig=contig, mask=mask)
            assert isinstance(filter_pass, da.Array)
            assert filter_pass.ndim == 1
            assert filter_pass.dtype == bool


@pytest.mark.parametrize("chunks", ["auto", "native"])
def test_snp_sites(chunks):

    ag3 = setup_ag3()

    # check can open the zarr directly
    root = ag3.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in contigs:
        assert contig in root

    # check access as dask arrays
    for contig in contigs:
        pos, ref, alt = ag3.snp_sites(contig=contig, chunks=chunks)
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

    # specific field
    pos = ag3.snp_sites(contig="3R", field="POS", chunks=chunks)
    assert isinstance(pos, da.Array)
    assert pos.ndim == 1
    assert pos.dtype == "i4"

    # apply site mask
    filter_pass = ag3.site_filters(contig="X", mask="gamb_colu_arab").compute()
    pos_pass = ag3.snp_sites(
        contig="X", field="POS", site_mask="gamb_colu_arab", chunks=chunks
    )
    assert isinstance(pos_pass, da.Array)
    assert pos_pass.ndim == 1
    assert pos_pass.dtype == "i4"
    assert pos_pass.shape[0] == np.count_nonzero(filter_pass)
    pos_pass, ref_pass, alt_pass = ag3.snp_sites(contig="X", site_mask="gamb_colu_arab")
    for d in pos_pass, ref_pass, alt_pass:
        assert isinstance(d, da.Array)
        assert d.shape[0] == np.count_nonzero(filter_pass)


@pytest.mark.parametrize("chunks", ["auto", "native"])
def test_snp_genotypes(chunks):

    ag3 = setup_ag3()

    # check can open the zarr directly
    root = ag3.open_snp_genotypes(sample_set="AG1000G-AO")
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in contigs:
        assert contig in root

    # check access as dask arrays
    sample_setss = (
        "v3",
        "v3_wild",
        "AG1000G-X",
        ["AG1000G-BF-A", "AG1000G-BF-B", "AG1000G-BF-C"],
    )

    for sample_sets in sample_setss:
        df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
        for contig in contigs:
            gt = ag3.snp_genotypes(
                contig=contig, sample_sets=sample_sets, chunks=chunks
            )
            assert isinstance(gt, da.Array)
            assert gt.ndim == 3
            assert gt.dtype == "i1"
            assert gt.shape[1] == len(df_samples)

    # specific fields
    x = ag3.snp_genotypes(contig="X", field="GT", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i1"
    x = ag3.snp_genotypes(contig="X", field="GQ", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "i2"
    x = ag3.snp_genotypes(contig="X", field="MQ", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "i2"
    x = ag3.snp_genotypes(contig="X", field="AD", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i2"

    # site mask
    filter_pass = ag3.site_filters(contig="X", mask="gamb_colu_arab").compute()
    df_samples = ag3.sample_metadata()
    gt_pass = ag3.snp_genotypes(contig="X", site_mask="gamb_colu_arab", chunks=chunks)
    assert isinstance(gt_pass, da.Array)
    assert gt_pass.ndim == 3
    assert gt_pass.dtype == "i1"
    assert gt_pass.shape[0] == np.count_nonzero(filter_pass)
    assert gt_pass.shape[1] == len(df_samples)
    assert gt_pass.shape[2] == 2


def test_genome():

    ag3 = setup_ag3()

    # test the open_genome() method to access as zarr
    genome = ag3.open_genome()
    assert isinstance(genome, zarr.hierarchy.Group)
    for contig in contigs:
        assert contig in genome
        assert genome[contig].dtype == "S1"

    # test the genome_sequence() method to access sequences
    for contig in contigs:
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


def test_is_accessible():

    ag3 = setup_ag3()
    # run a couple of tests
    tests = [("X", "gamb_colu_arab"), ("2R", "gamb_colu"), ("3L", "arab")]
    for contig, mask in tests:
        is_accessible = ag3.is_accessible(contig=contig, site_mask=mask)
        assert isinstance(is_accessible, np.ndarray)
        assert is_accessible.ndim == 1
        assert is_accessible.shape[0] == ag3.genome_sequence(contig).shape[0]


def test_cross_metadata():

    ag3 = setup_ag3()
    df_crosses = ag3.cross_metadata()
    assert isinstance(df_crosses, pd.DataFrame)
    expected_cols = ["cross", "sample_id", "father_id", "mother_id", "sex", "role"]
    assert df_crosses.columns.tolist() == expected_cols

    # check samples are in AG1000G-X
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-X", species_calls=None)
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
    for contig in "2R", "X":
        for site_mask in None, "gamb_colu_arab":
            pos = ag3.snp_sites(contig=contig, field="POS", site_mask=site_mask)
            for field in "codon_degeneracy", "seq_cls":
                d = ag3.site_annotations(
                    contig=contig, field=field, site_mask=site_mask
                )
                assert isinstance(d, da.Array)
                assert d.ndim == 1
                assert d.shape == pos.shape


@pytest.mark.parametrize("site_mask", [None, "gamb_colu_arab"])
@pytest.mark.parametrize("sample_sets", ["AG1000G-AO", "v3_wild"])
@pytest.mark.parametrize("contig", ["3L", "X"])
def test_snp_calls(sample_sets, contig, site_mask):

    ag3 = setup_ag3()

    ds = ag3.snp_calls(contig=contig, sample_sets=sample_sets, site_mask=site_mask)
    assert isinstance(ds, xarray.Dataset)

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
    pos = ag3.snp_sites(contig=contig, field="POS", site_mask=site_mask)
    n_variants = len(pos)
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    n_samples = len(df_samples)
    assert ds.dims["variants"] == n_variants
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 4

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
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

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xarray.DataArray)


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


def test_snp_allele_frequencies__no_samples():
    ag3 = setup_ag3()
    cohorts = {
        "bf_2050_col": "country == 'Burkina Faso' and year == 2050 and species == 'coluzzii'"
    }
    with pytest.raises(ValueError):
        _ = ag3.snp_allele_frequencies(
            transcript="AGAP009194-RA",
            cohorts=cohorts,
            site_mask="gamb_colu",
            sample_sets="v3_wild",
            drop_invariant=True,
        )


def test_snp_allele_frequencies__str_cohorts():
    ag3 = setup_ag3()
    cohorts = "admin1_month"
    universal_fields = [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
    ]
    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts=cohorts,
        cohorts_analysis="20210702",
        min_cohort_size=10,
        site_mask="gamb_colu",
        sample_sets="v3_wild",
        drop_invariant=True,
    )
    df_coh = ag3.sample_cohorts(sample_sets="v3_wild", cohorts_analysis="20210702")
    coh_nm = "cohort_" + cohorts
    all_uni = df_coh[coh_nm].dropna().unique().tolist()
    expected_fields = universal_fields + all_uni + ["max_af"]

    assert df.columns.tolist() == expected_fields
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (16639, 101)


def test_snp_allele_frequencies__dict_cohorts():
    ag3 = setup_ag3()
    cohorts = {
        "ke": "country == 'Kenya'",
        "bf_2012_col": "country == 'Burkina Faso' and year == 2012 and species == 'coluzzii'",
    }
    universal_fields = [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
        "pass_gamb_colu_arab",
        "pass_gamb_colu",
        "pass_arab",
    ]

    # test drop invariants
    df = ag3.snp_allele_frequencies(
        transcript="AGAP009194-RA",
        cohorts=cohorts,
        site_mask="gamb_colu",
        sample_sets="v3_wild",
        drop_invariant=True,
    )

    assert isinstance(df, pd.DataFrame)
    expected_fields = universal_fields + list(cohorts.keys()) + ["max_af"]
    assert df.columns.tolist() == expected_fields
    assert df.shape == (133, len(expected_fields))
    assert df.iloc[0].position == 28597653
    assert df.iloc[1].ref_allele == "A"
    assert df.iloc[2].alt_allele == "C"
    assert df.iloc[3].ke == 0
    assert df.iloc[4].bf_2012_col == pytest.approx(0.006097, abs=1e-6)
    assert df.iloc[4].max_af == pytest.approx(0.006097, abs=1e-6)
    # check invariant have been dropped
    assert df.max_af.min() > 0

    # test keep invariants
    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        cohorts=cohorts,
        site_mask="gamb_colu",
        sample_sets="v3_wild",
        drop_invariant=False,
    )
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == expected_fields
    assert df.shape == (132306, len(expected_fields))
    # check invariant positions are still present
    assert np.any(df.max_af == 0)


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-AO", ("AG1000G-AO", "AG1000G-UG"), "v3_wild"]
)
@pytest.mark.parametrize("contig", ["3L", "X"])
def test_cnv_hmm(sample_sets, contig):
    ag3 = setup_ag3()
    ds = ag3.cnv_hmm(contig=contig, sample_sets=sample_sets)
    assert isinstance(ds, xarray.Dataset)

    # check fields
    expected_data_vars = {
        "call_CN",
        "call_NormCov",
        "call_RawCov",
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
    n_variants = 1 + len(ag3.genome_sequence(contig=contig)) // 300
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    n_samples = len(df_samples)
    assert ds.dims["variants"] == n_variants
    assert ds.dims["samples"] == n_samples

    # check sample IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
        assert isinstance(x.data, da.Array)

        if f.startswith("variant_"):
            assert x.ndim, f == 1
            assert x.shape == (n_variants,)
            assert x.dims == ("variants",)
        elif f.startswith("call_"):
            assert x.ndim, f == 2
            assert x.dims == ("variants", "samples")
            assert x.shape == (n_variants, n_samples)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2R", "2L", "3R", "3L", "X")

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_CN"].sum(axis=1)
    assert isinstance(d2, xarray.DataArray)


@pytest.mark.parametrize("sample_set", ["AG1000G-AO", "AG1000G-UG", "AG1000G-X"])
@pytest.mark.parametrize("analysis", ["gamb_colu", "arab", "crosses"])
@pytest.mark.parametrize("contig", ["3L", "X"])
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
    assert isinstance(ds, xarray.Dataset)

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
    df_samples = ag3.sample_metadata(sample_sets=sample_set, species_calls=None)
    sample_id = pd.Series(ds["sample_id"].values)
    assert sample_id.isin(df_samples["sample_id"]).all()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
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

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_genotype"].sum(axis=1)
    assert isinstance(d2, xarray.DataArray)


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-AO", ("AG1000G-AO", "AG1000G-UG"), "v3_wild"]
)
@pytest.mark.parametrize("contig", ["2R", "3R", "X", "3L"])
def test_cnv_discordant_read_calls(sample_sets, contig):

    ag3 = setup_ag3()

    if contig == "3L":
        with pytest.raises(ValueError):
            ag3.cnv_discordant_read_calls(contig=contig, sample_sets=sample_sets)
        return

    ds = ag3.cnv_discordant_read_calls(contig=contig, sample_sets=sample_sets)
    assert isinstance(ds, xarray.Dataset)

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
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    n_samples = len(df_samples)
    assert ds.dims["samples"] == n_samples

    # check sample IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
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

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_genotype"].sum(axis=1)
    assert isinstance(d2, xarray.DataArray)


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


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-AO", ("AG1000G-TZ", "AG1000G-UG"), "v3_wild"]
)
@pytest.mark.parametrize("contig", ["2R", "X"])
def test_gene_cnv(contig, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(contig=contig, sample_sets=sample_sets)

    assert isinstance(ds, xarray.Dataset)

    # check fields
    expected_data_vars = {
        "CN_mode",
        "CN_mode_count",
        "gene_windows",
        "gene_name",
        "gene_strand",
    }
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "gene_id",
        "gene_start",
        "gene_end",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # check dimensions
    assert set(ds.dims) == {"samples", "genes"}

    # check dim lengths
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    n_samples = len(df_samples)
    assert ds.dims["samples"] == n_samples
    df_geneset = ag3.geneset()
    df_genes = df_geneset.query(f"type == 'gene' and contig == '{contig}'")
    n_genes = len(df_genes)
    assert ds.dims["genes"] == n_genes

    # check IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()
    assert ds["gene_id"].values.tolist() == df_genes["ID"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
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

    # check can setup computations
    d1 = ds["gene_start"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["CN_mode"].max(axis=1)
    assert isinstance(d2, xarray.DataArray)

    # sanity checks
    x = ds["gene_windows"].values
    y = ds["CN_mode_count"].values.max(axis=1)
    assert np.all(x >= y)
    z = ds["CN_mode"].values
    assert np.max(z) <= 12
    assert np.min(z) >= -1

    # check label-based indexing
    # pick a random gene and sample ID
    gene = random.choice(df_genes["ID"].tolist())
    sample = random.choice(df_samples["sample_id"].tolist())
    ds = ds.set_index(genes="gene_id", samples="sample_id")
    o = ds.sel(genes=gene)
    assert isinstance(o, xarray.Dataset)
    assert set(o.dims) == {"samples"}
    assert o.dims["samples"] == ds.dims["samples"]
    o = ds.sel(samples=sample)
    assert isinstance(o, xarray.Dataset)
    assert set(o.dims) == {"genes"}
    assert o.dims["genes"] == ds.dims["genes"]
    o = ds.sel(genes=gene, samples=sample)
    assert isinstance(o, xarray.Dataset)
    assert set(o.dims) == set()


@pytest.mark.parametrize("contig", ["2R", "X"])
@pytest.mark.parametrize(
    "cohorts",
    [
        {
            "ke": "country == 'Kenya'",
            "bf_2012_col": "country == 'Burkina Faso' and year == 2012 and species == 'coluzzii'",
        },
        {
            "bf_2050_col": "country == 'Burkina Faso' and year == 2050 and species == 'coluzzii'"
        },
        "admin1_month",
    ],
)
def test_gene_cnv_frequencies(contig, cohorts):

    universal_fields = ["contig", "start", "end", "strand", "Name", "description"]
    ag3 = setup_ag3()
    df_genes = ag3.geneset().query(f"type == 'gene' and contig == '{contig}'")
    if "bf_2050_col" in cohorts:
        with pytest.raises(ValueError):
            _ = ag3.gene_cnv_frequencies(
                contig=contig, sample_sets="v3_wild", cohorts=cohorts
            )
    else:
        df = ag3.gene_cnv_frequencies(
            contig=contig, sample_sets="v3_wild", cohorts=cohorts, min_cohort_size=0
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(df_genes)
        assert df.index.name == "ID"

        # sanity checks
        if isinstance(cohorts, dict):
            cohort_labels = cohorts.keys()
        if isinstance(cohorts, str):
            df_coh = ag3.sample_cohorts(
                sample_sets="v3_wild", cohorts_analysis="20210702"
            )
            coh_nm = "cohort_" + cohorts
            cohort_labels = list(df_coh[coh_nm].dropna().unique())

        suffixes = ["_amp", "_del"]
        cnv_freq_cols = [a + b for a in cohort_labels for b in suffixes]

        for f in cnv_freq_cols:
            x = df[f].values
            assert np.all(x >= 0)
            assert np.all(x <= 1)
        cnv_freq_col_pairs = list(zip(cnv_freq_cols[::2], cnv_freq_cols[1::2]))
        for fa, fd in cnv_freq_col_pairs:
            a = df[fa].values
            d = df[fd].values
            x = a + d
            assert np.all(x >= 0)
            assert np.all(x <= 1)
        expected_fields = universal_fields + cnv_freq_cols
        assert df.columns.tolist() == expected_fields


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-BF-A", ("AG1000G-TZ", "AG1000G-UG"), "v3", "v3_wild"]
)
@pytest.mark.parametrize("contig", ["2R", "X"])
@pytest.mark.parametrize("analysis", ["arab", "gamb_colu", "gamb_colu_arab"])
def test_haplotypes(sample_sets, contig, analysis):

    ag3 = setup_ag3()

    # check expected samples
    sample_query = None
    if analysis == "arab":
        sample_query = "species == 'arabiensis' and sample_set != 'AG1000G-X'"
    elif analysis == "gamb_colu":
        sample_query = "species in ['gambiae', 'coluzzii', 'intermediate_gambiae_coluzzii'] and sample_set != 'AG1000G-X'"
    elif analysis == "gamb_colu_arab":
        sample_query = "sample_set != 'AG1000G-X'"
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    expected_samples = set(df_samples.query(sample_query)["sample_id"].tolist())
    n_samples = len(expected_samples)

    # check if any samples
    if n_samples == 0:
        with pytest.raises(ValueError):
            # no samples, raise
            ag3.haplotypes(contig=contig, sample_sets=sample_sets, analysis=analysis)
        return

    ds = ag3.haplotypes(contig=contig, sample_sets=sample_sets, analysis=analysis)
    assert isinstance(ds, xarray.Dataset)

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
    samples = set(ds["sample_id"].values)
    assert samples == expected_samples

    # check dim lengths
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 2

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
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
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_genotype"].sum(axis=(1, 2))
    assert isinstance(d2, xarray.DataArray)


# test v3 sample sets
@pytest.mark.parametrize(
    "sample_sets", ["v3_wild", "v3", "AG1000G-UG", ["AG1000G-AO", "AG1000G-FR"]]
)
def test_sample_cohorts(sample_sets):
    expected_cols = (
        "sample_id",
        "cohort_admin1_year",
        "cohort_admin1_month",
        "cohort_admin2_year",
        "cohort_admin2_month",
    )

    ag3 = setup_ag3()
    df_coh = ag3.sample_cohorts(sample_sets=sample_sets, cohorts_analysis="20210702")
    df_meta = ag3.sample_metadata(sample_sets=sample_sets)

    assert tuple(df_coh.columns) == expected_cols
    assert len(df_coh) == len(df_meta)
    assert df_coh.sample_id.tolist() == df_meta.sample_id.tolist()
    if sample_sets == "AG1000G-UG":
        assert df_coh.sample_id[0] == "AC0007-C"
        assert df_coh.cohort_admin1_year[23] == "UG-E_2012_arab"
        assert df_coh.cohort_admin1_month[37] == "UG-E_2012_10_arab"
        assert df_coh.cohort_admin2_year[42] == "UG-E_Tororo_2012_arab"
        assert df_coh.cohort_admin2_month[49] == "UG-E_Tororo_2012_10_arab"
    if sample_sets == ["AG1000G-AO", "AG1000G-FR"]:
        assert df_coh.sample_id[0] == "AR0047-C"
        assert df_coh.sample_id[103] == "AP0017-Cx"
