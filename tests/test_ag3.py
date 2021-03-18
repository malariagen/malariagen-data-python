from malariagen_data import Ag3
import pandas
from pandas.testing import assert_frame_equal
import dask.array as da
import numpy as np
import zarr
import xarray
import pytest


gcs_url = "gs://vo_agam_release/"


expected_species = {
    "gambiae",
    "coluzzii",
    "arabiensis",
    "intermediate_arabiensis_gambiae",
    "intermediate_gambiae_coluzzii",
}


contigs = "2R", "2L", "3R", "3L", "X"


def test_sample_sets():

    ag3 = Ag3(gcs_url)
    df_sample_sets_v3 = ag3.sample_sets(release="v3")
    assert isinstance(df_sample_sets_v3, pandas.DataFrame)
    assert 28 == len(df_sample_sets_v3)
    assert ("sample_set", "sample_count", "release") == tuple(df_sample_sets_v3.columns)

    # test default is v3
    df_default = ag3.sample_sets()
    assert_frame_equal(df_sample_sets_v3, df_default)

    # try without trailing slash
    ag3 = Ag3(gcs_url[:-1])
    df_sample_sets_v3 = ag3.sample_sets(release="v3")
    assert isinstance(df_sample_sets_v3, pandas.DataFrame)
    assert 28 == len(df_sample_sets_v3)


def test_sample_metadata():

    ag3 = Ag3(gcs_url)
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

    ag3 = Ag3(gcs_url)
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

    ag3 = Ag3(gcs_url)

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


def test_snp_sites():

    ag3 = Ag3(gcs_url)

    # check can open the zarr directly
    root = ag3.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in contigs:
        assert contig in root

    # check access as dask arrays
    for contig in contigs:
        pos, ref, alt = ag3.snp_sites(contig=contig)
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
    pos = ag3.snp_sites(contig="3R", field="POS")
    assert isinstance(pos, da.Array)
    assert 1 == pos.ndim
    assert "i4" == pos.dtype

    # apply site mask
    filter_pass = ag3.site_filters(contig="X", mask="gamb_colu_arab").compute()
    pos_pass = ag3.snp_sites(contig="X", field="POS", site_mask="gamb_colu_arab")
    assert isinstance(pos_pass, da.Array)
    assert 1 == pos_pass.ndim
    assert "i4" == pos_pass.dtype
    assert np.count_nonzero(filter_pass) == pos_pass.shape[0]
    pos_pass, ref_pass, alt_pass = ag3.snp_sites(contig="X", site_mask="gamb_colu_arab")
    for d in pos_pass, ref_pass, alt_pass:
        assert isinstance(d, da.Array)
        assert np.count_nonzero(filter_pass) == d.shape[0]


def test_snp_genotypes():

    ag3 = Ag3(gcs_url)

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
            gt = ag3.snp_genotypes(contig=contig, sample_sets=sample_sets)
            assert isinstance(gt, da.Array)
            assert 3 == gt.ndim
            assert "i1" == gt.dtype
            assert len(df_samples) == gt.shape[1]

    # specific fields
    x = ag3.snp_genotypes(contig="X", field="GT")
    assert isinstance(x, da.Array)
    assert 3 == x.ndim
    assert "i1" == x.dtype
    x = ag3.snp_genotypes(contig="X", field="GQ")
    assert isinstance(x, da.Array)
    assert 2 == x.ndim
    assert "i2" == x.dtype
    x = ag3.snp_genotypes(contig="X", field="MQ")
    assert isinstance(x, da.Array)
    assert 2 == x.ndim
    assert "i2" == x.dtype
    x = ag3.snp_genotypes(contig="X", field="AD")
    assert isinstance(x, da.Array)
    assert 3 == x.ndim
    assert "i2" == x.dtype

    # site mask
    filter_pass = ag3.site_filters(contig="X", mask="gamb_colu_arab").compute()
    df_samples = ag3.sample_metadata()
    gt_pass = ag3.snp_genotypes(contig="X", site_mask="gamb_colu_arab")
    assert isinstance(gt_pass, da.Array)
    assert 3 == gt_pass.ndim
    assert "i1" == gt_pass.dtype
    assert np.count_nonzero(filter_pass) == gt_pass.shape[0]
    assert len(df_samples) == gt_pass.shape[1]
    assert 2 == gt_pass.shape[2]


def test_genome():

    ag3 = Ag3(gcs_url)

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

    ag3 = Ag3(gcs_url)

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

    ag3 = Ag3(gcs_url)
    # run a couple of tests
    tests = [("X", "gamb_colu_arab"), ("2R", "gamb_colu"), ("3L", "arab")]
    for contig, mask in tests:
        is_accessible = ag3.is_accessible(contig=contig, site_mask=mask)
        assert isinstance(is_accessible, np.ndarray)
        assert 1 == is_accessible.ndim
        assert ag3.genome_sequence(contig).shape[0] == is_accessible.shape[0]


def test_cross_metadata():

    ag3 = Ag3(gcs_url)
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

    ag3 = Ag3(gcs_url)

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
    for contig in "2R", "X", "3L":
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

    ag3 = Ag3(gcs_url)

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
