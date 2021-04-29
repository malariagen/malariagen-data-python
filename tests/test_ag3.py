import dask.array as da
import numpy as np
import pandas
import pytest
import xarray
import zarr
from pandas.testing import assert_frame_equal

from malariagen_data import Ag3

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
    assert isinstance(df_sample_sets_v3, pandas.DataFrame)
    assert 28 == len(df_sample_sets_v3)
    assert ("sample_set", "sample_count", "release") == tuple(df_sample_sets_v3.columns)

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
    assert expected_cols == tuple(df_samples_v3.columns)
    expected_len = df_sample_sets_v3["sample_count"].sum()
    assert expected_len == len(df_samples_v3)

    # v3_wild
    df_samples_v3_wild = ag3.sample_metadata(sample_sets="v3_wild", species_calls=None)
    assert expected_cols == tuple(df_samples_v3_wild.columns)
    expected_len = df_sample_sets_v3.query("sample_set != 'AG1000G-X'")[
        "sample_count"
    ].sum()
    assert expected_len == len(df_samples_v3_wild)

    # single sample set
    df_samples_x = ag3.sample_metadata(sample_sets="AG1000G-X", species_calls=None)
    assert expected_cols == tuple(df_samples_x.columns)
    expected_len = df_sample_sets_v3.query("sample_set == 'AG1000G-X'")[
        "sample_count"
    ].sum()
    assert expected_len == len(df_samples_x)

    # multiple sample sets
    sample_sets = ["AG1000G-BF-A", "AG1000G-BF-B", "AG1000G-BF-C"]
    df_samples_bf = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    assert expected_cols == tuple(df_samples_bf)
    loc_sample_sets = df_sample_sets_v3["sample_set"].isin(sample_sets)
    expected_len = df_sample_sets_v3.loc[loc_sample_sets]["sample_count"].sum()
    assert expected_len == len(df_samples_bf)

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
    assert expected_cols + aim_cols == tuple(df_samples_aim.columns)
    assert len(df_samples_v3_wild) == len(df_samples_aim)
    assert expected_species == set(df_samples_aim["species"])

    # AIM species calls, explicit
    df_samples_aim = ag3.sample_metadata(species_calls=("20200422", "aim"))
    assert expected_cols + aim_cols == tuple(df_samples_aim.columns)
    assert len(df_samples_v3_wild) == len(df_samples_aim)
    assert expected_species == set(df_samples_aim["species"])

    pca_cols = (
        "PC1",
        "PC2",
        "species_gambcolu_arabiensis",
        "species_gambiae_coluzzii",
        "species",
    )

    # PCA species calls
    df_samples_pca = ag3.sample_metadata(species_calls=("20200422", "pca"))
    assert expected_cols + pca_cols == tuple(df_samples_pca.columns)
    assert len(df_samples_v3_wild) == len(df_samples_pca)
    assert set() == set(df_samples_pca["species"]).difference(expected_species)


def test_species_calls():

    ag3 = setup_ag3()
    sample_sets = ag3.sample_sets(release="v3")["sample_set"].tolist()

    for s in sample_sets:
        for method in "aim", "pca":
            df_samples = ag3.sample_metadata(sample_sets=s, species_calls=None)
            df_species = ag3.species_calls(sample_sets=s, method=method)
            assert len(df_samples) == len(df_species)
            if s == "AG1000G-X":
                # no species calls
                assert df_species["species"].isna().all()
            else:
                assert not df_species["species"].isna().any()
                assert set() == set(df_species["species"]).difference(expected_species)


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
            assert 1 == filter_pass.ndim
            assert bool == filter_pass.dtype


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
        assert 1 == pos.ndim
        assert "i4" == pos.dtype
        assert isinstance(ref, da.Array)
        assert 1 == ref.ndim
        assert "S1" == ref.dtype
        assert isinstance(alt, da.Array)
        assert 2 == alt.ndim
        assert "S1" == alt.dtype
        assert pos.shape[0] == ref.shape[0] == alt.shape[0]

    # specific field
    pos = ag3.snp_sites(contig="3R", field="POS", chunks=chunks)
    assert isinstance(pos, da.Array)
    assert 1 == pos.ndim
    assert "i4" == pos.dtype

    # apply site mask
    filter_pass = ag3.site_filters(contig="X", mask="gamb_colu_arab").compute()
    pos_pass = ag3.snp_sites(
        contig="X", field="POS", site_mask="gamb_colu_arab", chunks=chunks
    )
    assert isinstance(pos_pass, da.Array)
    assert 1 == pos_pass.ndim
    assert "i4" == pos_pass.dtype
    assert np.count_nonzero(filter_pass) == pos_pass.shape[0]
    pos_pass, ref_pass, alt_pass = ag3.snp_sites(contig="X", site_mask="gamb_colu_arab")
    for d in pos_pass, ref_pass, alt_pass:
        assert isinstance(d, da.Array)
        assert np.count_nonzero(filter_pass) == d.shape[0]


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
            assert 3 == gt.ndim
            assert "i1" == gt.dtype
            assert len(df_samples) == gt.shape[1]

    # specific fields
    x = ag3.snp_genotypes(contig="X", field="GT", chunks=chunks)
    assert isinstance(x, da.Array)
    assert 3 == x.ndim
    assert "i1" == x.dtype
    x = ag3.snp_genotypes(contig="X", field="GQ", chunks=chunks)
    assert isinstance(x, da.Array)
    assert 2 == x.ndim
    assert "i2" == x.dtype
    x = ag3.snp_genotypes(contig="X", field="MQ", chunks=chunks)
    assert isinstance(x, da.Array)
    assert 2 == x.ndim
    assert "i2" == x.dtype
    x = ag3.snp_genotypes(contig="X", field="AD", chunks=chunks)
    assert isinstance(x, da.Array)
    assert 3 == x.ndim
    assert "i2" == x.dtype

    # site mask
    filter_pass = ag3.site_filters(contig="X", mask="gamb_colu_arab").compute()
    df_samples = ag3.sample_metadata()
    gt_pass = ag3.snp_genotypes(contig="X", site_mask="gamb_colu_arab", chunks=chunks)
    assert isinstance(gt_pass, da.Array)
    assert 3 == gt_pass.ndim
    assert "i1" == gt_pass.dtype
    assert np.count_nonzero(filter_pass) == gt_pass.shape[0]
    assert len(df_samples) == gt_pass.shape[1]
    assert 2 == gt_pass.shape[2]


def test_genome():

    ag3 = setup_ag3()

    # test the open_genome() method to access as zarr
    genome = ag3.open_genome()
    assert isinstance(genome, zarr.hierarchy.Group)
    for contig in contigs:
        assert contig in genome
        assert "S1" == genome[contig].dtype

    # test the genome_sequence() method to access sequences
    for contig in contigs:
        seq = ag3.genome_sequence(contig)
        assert isinstance(seq, da.Array)
        assert "S1" == seq.dtype


def test_geneset():

    ag3 = setup_ag3()

    # default
    df = ag3.geneset()
    assert isinstance(df, pandas.DataFrame)
    gff3_cols = [
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
    ]
    expected_cols = gff3_cols + ["ID", "Parent", "Name"]
    assert expected_cols == df.columns.tolist()

    # don't unpack attributes
    df = ag3.geneset(attributes=None)
    assert isinstance(df, pandas.DataFrame)
    expected_cols = gff3_cols + ["attributes"]
    assert expected_cols == df.columns.tolist()


def test_is_accessible():

    ag3 = setup_ag3()
    # run a couple of tests
    tests = [("X", "gamb_colu_arab"), ("2R", "gamb_colu"), ("3L", "arab")]
    for contig, mask in tests:
        is_accessible = ag3.is_accessible(contig=contig, site_mask=mask)
        assert isinstance(is_accessible, np.ndarray)
        assert 1 == is_accessible.ndim
        assert ag3.genome_sequence(contig).shape[0] == is_accessible.shape[0]


def test_cross_metadata():

    ag3 = setup_ag3()
    df_crosses = ag3.cross_metadata()
    assert isinstance(df_crosses, pandas.DataFrame)
    expected_cols = ["cross", "sample_id", "father_id", "mother_id", "sex", "role"]
    assert expected_cols == df_crosses.columns.tolist()

    # check samples are in AG1000G-X
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-X", species_calls=None)
    assert set(df_samples["sample_id"]) == set(df_crosses["sample_id"])

    # check values
    expected_role_values = ["parent", "progeny"]
    assert expected_role_values == df_crosses["role"].unique().tolist()
    expected_sex_values = ["F", "M"]
    assert expected_sex_values == df_crosses["sex"].unique().tolist()


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
                assert 1 == d.ndim
                assert pos.shape == d.shape


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
    assert expected_data_vars == set(ds.data_vars)

    expected_coords = {
        "variant_contig",
        "variant_position",
        "sample_id",
    }
    assert expected_coords == set(ds.coords)

    # check dimensions
    assert {"alleles", "ploidy", "samples", "variants"} == set(ds.dims)

    # check dim lengths
    pos = ag3.snp_sites(contig=contig, field="POS", site_mask=site_mask)
    n_variants = len(pos)
    df_samples = ag3.sample_metadata(sample_sets=sample_sets, species_calls=None)
    n_samples = len(df_samples)
    assert n_variants == ds.dims["variants"]
    assert n_samples == ds.dims["samples"]
    assert 2 == ds.dims["ploidy"]
    assert 4 == ds.dims["alleles"]

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert 2 == x.ndim, f
            assert (n_variants, 4) == x.shape
            assert ("variants", "alleles") == x.dims
        elif f.startswith("variant_"):
            assert 1 == x.ndim, f
            assert (n_variants,) == x.shape
            assert ("variants",) == x.dims
        elif f in {"call_genotype", "call_genotype_mask"}:
            assert 3 == x.ndim
            assert ("variants", "samples", "ploidy") == x.dims
            assert (n_variants, n_samples, 2) == x.shape
        elif f == "call_AD":
            assert 3 == x.ndim
            assert ("variants", "samples", "alleles") == x.dims
            assert (n_variants, n_samples, 4) == x.shape
        elif f.startswith("call_"):
            assert 2 == x.ndim, f
            assert ("variants", "samples") == x.dims
            assert (n_variants, n_samples) == x.shape
        elif f.startswith("sample_"):
            assert 1 == x.ndim
            assert ("samples",) == x.dims
            assert (n_samples,) == x.shape

    # check attributes
    assert "contigs" in ds.attrs
    assert ("2R", "2L", "3R", "3L", "X") == ds.attrs["contigs"]

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
        "position",
        "ref_allele",
        "alt_allele",
        "effect",
        "impact",
        "ref_codon",
        "alt_codon",
        "aa_pos",
        "ref_aa",
        "alt_aa",
        "aa_change",
    ]

    df = ag3.snp_effects(gste2, site_mask)
    assert isinstance(df, pandas.DataFrame)
    assert expected_fields == df.columns.tolist()

    # reverse strand gene
    assert df.shape == (2838, 11)
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
    df = ag3.snp_effects(gste6, site_mask)
    assert isinstance(df, pandas.DataFrame)
    assert expected_fields == df.columns.tolist()
    assert df.shape == (2829, 11)

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
    df = ag3.snp_effects(utrintron5, site_mask)
    assert isinstance(df, pandas.DataFrame)
    assert expected_fields == df.columns.tolist()
    assert df.shape == (7686, 11)
    assert df.loc[180].effect == "SPLICE_CORE"
    assert df.loc[198].effect == "SPLICE_REGION"
    assert df.loc[202].effect == "INTRONIC"

    # check 3' utr intron
    utrintron3 = "AGAP000689-RA"
    df = ag3.snp_effects(utrintron3, site_mask)
    assert isinstance(df, pandas.DataFrame)
    assert expected_fields == df.columns.tolist()
    assert df.shape == (5397, 11)
    assert df.loc[646].effect == "SPLICE_CORE"
    assert df.loc[652].effect == "SPLICE_REGION"
    assert df.loc[674].effect == "INTRONIC"

def test_get_only_transcript_effects():
    from malariagen_data import veff
    ag3 = setup_ag3()
    gste2 = "AGAP009194-RA"
    d = {'position': [28598166,], 'ref_allele': ["A",], 'alt_allele': ["G",]}
    variants = pandas.DataFrame(data=d)
    ann = veff.Annotator(genome=ag3.open_genome(), geneset=ag3.geneset())
    df = ann.get_only_transcript_effects(gste2, variants)
    assert df.iloc[0].effect == "NON_SYNONYMOUS_CODING"



def test_snp_allele_frequencies():
    ag3 = setup_ag3()
    populations = {
        "ke": "country == 'Kenya'",
        "bf_2012_col": "country == 'Burkina Faso' and year == 2012 and species == 'coluzzii'",
    }
    expected_fields = [
        "position",
        "ref_allele",
        "alt_allele",
        "ke",
        "bf_2012_col",
        "maximum",
    ]
    # drop invariants
    df = ag3.snp_allele_frequencies(
        transcript="AGAP009194-RA",
        populations=populations,
        site_mask="gamb_colu",
        sample_sets="v3_wild",
        drop_invariant=True,
    )

    assert isinstance(df, pandas.DataFrame)
    assert expected_fields == df.columns.tolist()
    assert df.shape == (133, 6)
    assert df.loc[4].position == 28597653
    assert df.loc[5].ref_allele == "A"
    assert df.loc[13].alt_allele == "C"
    assert df.loc[16].ke == 0
    assert df.loc[22].bf_2012_col == pytest.approx(0.006097, abs=1e-6)
    assert df.loc[39].maximum == pytest.approx(0.006097, abs=1e-6)
    # check invariant have been dropped
    assert df.maximum.min() > 0

    populations = {
        "gm": "country == 'Gambia, The'",
        "mz": "country == 'Mozambique' and year == 2004",
    }
    expected_fields = [
        "position",
        "ref_allele",
        "alt_allele",
        "gm",
        "mz",
        "maximum",
    ]
    # keep invariants
    df = ag3.snp_allele_frequencies(
        transcript="AGAP004707-RD",
        populations=populations,
        site_mask="gamb_colu",
        sample_sets="v3_wild",
        drop_invariant=False,
    )

    assert isinstance(df, pandas.DataFrame)
    assert expected_fields == df.columns.tolist()
    assert df.shape == (132306, 6)
    assert df.loc[0].position == 2358158
    assert df.loc[1].ref_allele == "A"
    assert df.loc[2].alt_allele == "G"
    assert df.loc[3].gm == 0.0
    assert df.loc[4].mz == 0.0
    assert df.loc[72].maximum == pytest.approx(0.001792, abs=1e-6)
    # check invariant positions are still present
    assert np.any(df.maximum == 0)
