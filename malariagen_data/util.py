from collections.abc import Mapping
from enum import Enum
from urllib.parse import unquote_plus

import dask.array as da
import numpy as np
import pandas
import xarray as xr


def gff3_parse_attributes(attributes_string):
    """Parse a string of GFF3 attributes ('key=value' pairs delimited by ';')
    and return a dictionary."""

    attributes = dict()
    fields = attributes_string.split(";")
    for f in fields:
        if "=" in f:
            key, value = f.split("=")
            key = unquote_plus(key).strip()
            value = unquote_plus(value.strip())
            attributes[key] = value
        elif len(f) > 0:
            # not strictly kosher, treat as a flag
            attributes[unquote_plus(f).strip()] = True
    return attributes


gff3_cols = (
    "contig",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
    "attributes",
)


def read_gff3(buf, compression="gzip"):

    # read as dataframe
    df = pandas.read_csv(
        buf,
        sep="\t",
        comment="#",
        names=gff3_cols,
        na_values=["", "."],
        compression=compression,
    )

    # parse attributes
    df["attributes"] = df["attributes"].apply(gff3_parse_attributes)

    return df


def unpack_gff3_attributes(df, attributes):

    df = df.copy()

    if attributes == "*":
        # discover all attribute keys
        attributes = set()
        for a in df["attributes"]:
            attributes.update(a.keys())
        attributes = sorted(attributes)

    # unpack attributes into columns
    for key in attributes:
        df[key] = df["attributes"].apply(lambda v: v.get(key, np.nan))
    del df["attributes"]

    return df


class SafeStore(Mapping):
    def __init__(self, store):
        self.store = store

    def __getitem__(self, key):
        try:
            return self.store[key]
        except KeyError as e:
            # always raise a runtime error to ensure zarr propagates the exception
            raise RuntimeError(e)

    def __contains__(self, key):
        return key in self.store

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)


class SiteClass(Enum):
    UPSTREAM = 1
    DOWNSTREAM = 2
    UTR5 = 3
    UTR3 = 4
    CDS_FIRST = 5
    CDS_MID = 6
    CDS_LAST = 7
    INTRON_FIRST = 8
    INTRON_MID = 9
    INTRON_LAST = 10


def from_zarr(z, inline_array, chunks="auto"):
    """Utility function for turning a zarr array into a dask array.

    N.B., dask does have it's own from_zarr() function but we roll
    our own here to get a little more control.

    """
    if chunks == "native" or z.dtype == object:
        # N.B., dask does not support "auto" chunks for arrays with object dtype
        chunks = z.chunks
    kwargs = dict(chunks=chunks, fancy=False, lock=False, inline_array=inline_array)
    try:
        d = da.from_array(z, **kwargs)
    except TypeError:
        # only later versions of dask support inline_array argument
        del kwargs["inline_array"]
        d = da.from_array(z, **kwargs)
    return d


def dask_compress_dataset(ds, indexer, dim):
    """Temporary workaround for memory issues when attempting to
    index an xarray dataset with a Boolean array.

    See also: https://github.com/pydata/xarray/issues/5054

    Parameters
    ----------
    ds : xarray.Dataset
    indexer : str
    dim : str

    Returns
    -------
    xarray.Dataset

    """
    if isinstance(indexer, str):
        indexer = ds[indexer].data

    # sanity checks
    assert isinstance(indexer, da.Array)
    assert indexer.ndim == 1
    assert indexer.dtype == bool
    assert indexer.shape[0] == ds.dims[dim]

    coords = dict()
    for k in ds.coords:
        a = ds[k]
        v = _dask_compress_dataarray(a, indexer, dim)
        coords[k] = (a.dims, v)

    data_vars = dict()
    for k in ds.data_vars:
        a = ds[k]
        v = _dask_compress_dataarray(a, indexer, dim)
        data_vars[k] = (a.dims, v)

    attrs = ds.attrs.copy()

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _dask_compress_dataarray(a, indexer, dim):
    try:
        # find the axis for the given dimension
        axis = a.dims.index(dim)

    except ValueError:
        # array doesn't have the given dimension, return as-is
        v = a.data

    else:
        # apply the indexing operation
        v = dask_compress(indexer, a.data, axis)

    return v


def dask_compress(indexer, data, axis):
    """Wrapper for dask.array.compress() which computes chunk sizes faster."""

    # sanity checks
    assert isinstance(data, da.Array)
    assert isinstance(indexer, da.Array)
    assert isinstance(axis, int)
    assert indexer.shape[0] == data.shape[axis]
    old_chunks = data.chunks
    axis_old_chunks = old_chunks[axis]
    axis_n_chunks = len(axis_old_chunks)

    # apply the indexing operation
    v = da.compress(indexer, data, axis=axis)

    # need to compute chunks sizes in order to know dimension sizes;
    # would normally do v.compute_chunk_sizes() but that is slow for
    # multidimensional arrays, so hack something more efficient

    axis_new_chunks = tuple(
        indexer.rechunk((axis_old_chunks,))
        .map_blocks(
            lambda b: np.sum(b, keepdims=True),
            chunks=((1,) * axis_n_chunks,),
        )
        .compute()
    )
    new_chunks = tuple(
        [axis_new_chunks if i == axis else c for i, c in enumerate(old_chunks)]
    )
    v._chunks = new_chunks

    return v
