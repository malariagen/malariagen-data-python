import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray
import zarr

from malariagen_data import Amin1


def setup_amin1(url="simplecache::gs://vo_amin_release/", **kwargs):
    if url is None:
        # test default URL
        return Amin1(**kwargs)
    if url.startswith("simplecache::"):
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Amin1(url, **kwargs)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "gs://vo_amin_release/",
        "gcs://vo_amin_release/",
        "gs://vo_amin_release",
        "gcs://vo_amin_release",
        "simplecache::gs://vo_amin_release/",
        "simplecache::gcs://vo_amin_release/",
    ],
)
def test_sample_metadata(url):
    amin1 = setup_amin1(url)

    expected_cols = (
        "sample_id",
        "original_sample_id",
        "sanger_sample_id",
        "partner_sample_id",
        "contributor",
        "country",
        "location",
        "year",
        "month",
        "latitude",
        "longitude",
        "season",
        "PCA_cohort",
        "cohort",
        "subsampled_cohort",
    )

    df_samples = amin1.sample_metadata()
    assert len(df_samples) == 302
    assert tuple(df_samples.columns) == expected_cols


def test_genome():
    amin1 = setup_amin1()

    # test the open_genome() method to access as zarr
    genome = amin1.open_genome()
    assert isinstance(genome, zarr.hierarchy.Group)
    for contig in genome:
        assert contig.startswith("KB66"), contig
        assert genome[contig].dtype == "S1", contig
        # test the genome_sequence() method to access sequences
        seq = amin1.genome_sequence(contig)
        assert isinstance(seq, da.Array)
        assert seq.dtype == "S1"

    # test the contigs property
    assert len(amin1.contigs) == 40  # only 40 largest included
    assert set(amin1.contigs) - set(genome) == set()


def test_genome_features():
    amin1 = setup_amin1()

    # default
    df = amin1.genome_features()
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
    df = amin1.genome_features(attributes=None)
    assert isinstance(df, pd.DataFrame)
    expected_cols = gff3_cols + ["attributes"]
    assert df.columns.tolist() == expected_cols


@pytest.mark.parametrize("site_mask", [False, True])
@pytest.mark.parametrize(
    "region",
    [
        "KB663610",
        "KB663622",
        ["KB663610", "KB663611", "KB663622"],
        "KB663610:100000-200000",
        "AMIN002150",
    ],
)
def test_snp_calls(region, site_mask):
    amin1 = setup_amin1()

    ds = amin1.snp_calls(region=region, site_mask=site_mask)
    assert isinstance(ds, xarray.Dataset)

    # check fields
    expected_data_vars = {
        "variant_allele",
        "variant_filter_pass",
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
    df_samples = amin1.sample_metadata()
    n_samples = len(df_samples)
    n_variants = ds.dims["variants"]
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

    # check variant_filter_pass
    filter_pass = ds["variant_filter_pass"].values
    n_pass = np.count_nonzero(filter_pass)
    if site_mask:
        # variant filter has already been applied
        assert n_pass == n_variants
    else:
        assert n_pass < n_variants

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == amin1.contigs

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xarray.DataArray)
