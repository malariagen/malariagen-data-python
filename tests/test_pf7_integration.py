import dask.array as da
import pytest

from malariagen_data.pf7 import Pf7


def setup_pf7(url="simplecache::gs://pf7_staging/", **storage_kwargs):
    if url.startswith("simplecache::"):
        storage_kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Pf7(url, **storage_kwargs)


@pytest.mark.parametrize(
    "url",
    [
        "gs://pf7_staging/",
        "gcs://pf7_staging/",
        "gs://pf7_staging",
        "gcs://pf7_staging",
        "simplecache::gs://pf7_staging/",
        "simplecache::gcs://pf7_staging/",
    ],
)
def test_sample_metadata(url):

    pf7 = setup_pf7(url)
    df_samples = pf7.sample_metadata()

    expected_cols = (
        "Sample",
        "Study",
        "Country",
        "Admin level 1",
        "Country latitude",
        "Country longitude",
        "Admin level 1 latitude",
        "Admin level 1 longitude",
        "Year",
        "ENA",
        "All samples same case",
        "Population",
        "% callable",
        "QC pass",
        "Exclusion reason",
        "Sample type",
        "Sample was in Pf6",
    )

    assert tuple(df_samples.columns) == expected_cols
    expected_len = 20864
    assert len(df_samples) == expected_len


@pytest.mark.parametrize("chunks", ["auto", "native"])
def test_variants(chunks):

    pf7 = setup_pf7()

    # check can open the zarr directly
    root = pf7.variants()
    assert isinstance(root, tuple)

    # check access as dask arrays
    chrom, pos = pf7.variants(chunks=chunks)
    assert isinstance(chrom, da.Array)
    assert chrom.ndim == 1
    assert chrom.dtype == "O"
    assert isinstance(pos, da.Array)
    assert pos.ndim == 1
    assert pos.dtype == "i4"
    assert chrom.shape[0] == pos.shape[0]

    # specific field
    pos = pf7.variants(field="POS", chunks=chunks)
    assert isinstance(pos, da.Array)
    assert pos.ndim == 1
    assert pos.dtype == "i4"


@pytest.mark.parametrize("chunks", ["auto", "native"])
def test_calldata(chunks):

    pf7 = setup_pf7()

    # check can open the zarr directly
    root = pf7.calldata()
    assert isinstance(root, da.Array)

    df_samples = pf7.sample_metadata()
    gt = pf7.calldata(chunks=chunks)
    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert gt.shape[1] == len(df_samples)

    # specific fields
    x = pf7.calldata(field="GT", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i1"
    x = pf7.calldata(field="GQ", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "int8"
    x = pf7.calldata(field="AD", chunks=chunks)
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "int16"
