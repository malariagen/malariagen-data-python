import hashlib
import json
import logging
import re
import sys
import warnings
from collections import Counter
from enum import Enum
from math import prod
from functools import wraps
from inspect import getcallargs
from textwrap import dedent, fill
from typing import IO, Dict, Hashable, List, Mapping, Optional, Tuple, Union, Callable
from urllib.parse import unquote_plus
from numpy.testing import assert_allclose, assert_array_equal

try:
    from google import colab  # type: ignore
except ImportError:
    colab = None

import allel  # type: ignore
import dask.array as da
from dask.utils import parse_bytes
import numba  # type: ignore
import numpy as np
import pandas
import pandas as pd
import plotly.express as px  # type: ignore
import typeguard
import xarray as xr
import zarr  # type: ignore

# zarr >= 2.11.0
from zarr.storage import BaseStore  # type: ignore
from fsspec.core import url_to_fs  # type: ignore
from fsspec.mapping import FSMap  # type: ignore
from numpydoc_decorator.impl import humanize_type  # type: ignore
from typing_extensions import TypeAlias, get_type_hints

DIM_VARIANT = "variants"
DIM_ALLELE = "alleles"
DIM_SAMPLE = "samples"
DIM_PLOIDY = "ploidy"


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


def unpack_gff3_attributes(df: pd.DataFrame, attributes: Tuple[str, ...]):
    df = df.copy()

    # discover all attribute keys
    all_attributes = set()
    for a in df["attributes"]:
        all_attributes.update(a.keys())
    all_attributes_sorted = tuple(sorted(all_attributes))

    # handle request for all attributes
    if attributes == ("*",):
        attributes = all_attributes_sorted

    # unpack attributes into columns
    for key in attributes:
        if key not in all_attributes_sorted:
            raise ValueError(
                f"'{key}' not in attributes set. Options {all_attributes_sorted}"
            )
        df[key] = df["attributes"].apply(lambda v: v.get(key, np.nan))
    del df["attributes"]

    return df


class SafeStore(BaseStore):
    """This class wraps any zarr store and ensures that missing chunks
    will not get automatically filled but will raise an exception. There
    should be no missing chunks in any of the datasets we host."""

    def __init__(self, store):
        self._store = store

    def __getitem__(self, key):
        try:
            return self._store[key]
        except KeyError as e:
            # Raise a different error to ensure zarr propagates the exception,
            # rather than filling.
            raise FileNotFoundError(e)

    def __getattr__(self, attr):
        if attr == "__setstate__":
            # Special method called during unpickling, don't pass through.
            raise AttributeError(attr)
        # Pass through all other attribute access to the wrapped store.
        return getattr(self._store, attr)

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __setitem__(self, item):
        raise NotImplementedError

    def __delitem__(self, item):
        raise NotImplementedError


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


zarr_chunks_type: TypeAlias = Tuple[int, ...]

dask_chunks_type: TypeAlias = Union[
    int,
    str,
    Tuple[Union[int, str], ...],
]

chunks_param_type: TypeAlias = Union[
    dask_chunks_type,
    Callable[[zarr_chunks_type], dask_chunks_type],
]


def da_from_zarr(
    z: zarr.core.Array,
    inline_array: bool,
    chunks: chunks_param_type,
) -> da.Array:
    """Utility function for turning a zarr array into a dask array.

    N.B., dask does have its own from_zarr() function, but we roll
    our own here to get a little more control.

    """

    # Some function of the native chunk sizes.
    if callable(chunks):
        dask_chunks: dask_chunks_type = chunks(z.chunks)

    # Match the zarr chunk size.
    # N.B., dask does not support "auto" chunks for arrays with object dtype.
    elif chunks == "native" or z.dtype == object:
        dask_chunks = z.chunks

    # Let dask auto-size chunks. This generally does not work well with
    # our datasets, but support this option in case someone wants to try
    # it.
    elif chunks == "auto":
        dask_chunks = chunks

    # Auto-size chunks but only for arrays with more than one dimension. This
    # seems to lead to unexpected memory usage in some scenarios, and so is not
    # the recommended option, but we'll leave these options here for now in
    # case further experiments are useful.
    elif chunks == "ndauto":
        if len(z.chunks) > 1:
            # Auto-size all dimensions.
            dask_chunks = "auto"
        else:
            dask_chunks = z.chunks
    elif chunks == "ndauto0":
        if len(z.chunks) > 1:
            # Auto-size first dimension.
            dask_chunks = ("auto",) + z.chunks[1:]
        else:
            dask_chunks = z.chunks
    elif chunks == "ndauto1":
        if len(z.chunks) > 1:
            # Auto-size second dimension.
            dask_chunks = (z.chunks[0], "auto") + z.chunks[2:]
        else:
            dask_chunks = z.chunks
    elif chunks == "ndauto01":
        if len(z.chunks) > 1:
            # Auto-size first and second dimensions.
            dask_chunks = ("auto", "auto") + z.chunks[2:]
        else:
            dask_chunks = z.chunks

    # Resize chunks to a specific size in memory.
    #
    # N.B., only resize chunks in arrays with more than one dimension,
    # because resizing the one-dimensional arrays according to the same
    # size may lead to poor performance with our datasets.
    #
    # Also, resize along the first dimension only. Again, this is something
    # that may work well for our datasets.
    #
    # Note that dask also supports this kind of argument, and so we could
    # just pass this through. However, some experiments have found this
    # leads to excessive memory usage. So we will control this behaviour
    # ourselves here and make sure dask chunk sizes are always a multiple of the
    # underlying zarr chunk sizes.
    elif isinstance(chunks, str):
        if len(z.chunks) > 1:
            dask_chunk_nbytes = parse_bytes(chunks)
            zarr_chunk_nbytes = prod(z.chunks) * z.dtype.itemsize
            factor = dask_chunk_nbytes // zarr_chunk_nbytes
            if factor > 1:
                dask_chunks = ((z.chunks[0] * factor),) + z.chunks[1:]
            else:
                dask_chunks = z.chunks
        else:
            dask_chunks = z.chunks

    # Pass through argument as-is to dask.
    else:
        dask_chunks = chunks

    kwargs = dict(
        inline_array=inline_array,
        chunks=dask_chunks,
        fancy=True,
        lock=False,
        asarray=True,
    )
    try:
        d = da.from_array(z, **kwargs)
    except TypeError:
        # only later versions of dask support inline_array argument
        del kwargs["inline_array"]
        d = da.from_array(z, **kwargs)
    return d


def dask_compress_dataset(ds, indexer, dim):
    """Temporary workaround for memory issues when attempting to
    index a xarray dataset with a Boolean array.

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
    assert indexer.ndim == 1
    assert indexer.dtype == bool
    assert indexer.shape[0] == ds.sizes[dim]

    if isinstance(indexer, da.Array):
        # temporarily compute the indexer once, to avoid multiple reads from
        # the underlying data
        indexer_computed = indexer.compute()
    else:
        assert isinstance(indexer, np.ndarray)
        indexer_computed = indexer

    coords = dict()
    for k in ds.coords:
        a = ds[k]
        v = _dask_compress_dataarray(a, indexer, indexer_computed, dim)
        coords[k] = (a.dims, v)

    data_vars = dict()
    for k in ds.data_vars:
        a = ds[k]
        v = _dask_compress_dataarray(a, indexer, indexer_computed, dim)
        data_vars[k] = (a.dims, v)

    attrs = ds.attrs.copy()

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _dask_compress_dataarray(a, indexer, indexer_computed, dim):
    try:
        # find the axis for the given dimension
        axis = a.dims.index(dim)

    except ValueError:
        # array doesn't have the given dimension, return as-is
        v = a.data

    else:
        # apply the indexing operation
        data = a.data
        if isinstance(data, da.Array):
            v = da_compress(
                indexer=indexer,
                data=a.data,
                axis=axis,
                indexer_computed=indexer_computed,
            )
        else:
            v = np.compress(indexer_computed, data, axis=axis)

    return v


def da_compress(
    indexer: da.Array | np.ndarray,
    data: da.Array,
    axis: int,
    indexer_computed: Optional[np.ndarray] = None,
):
    """Wrapper for dask.array.compress() which computes chunk sizes faster."""

    # Sanity checks.
    assert indexer.ndim == 1
    assert indexer.dtype == bool
    assert indexer.shape[0] == data.shape[axis]

    # Useful variables.
    old_chunks = data.chunks
    axis_old_chunks = old_chunks[axis]

    # Load the indexer temporarily for chunk size computations.
    if indexer_computed is None:
        assert isinstance(indexer, da.Array)
        indexer_computed = indexer.compute()

    # Ensure indexer and data are chunked in the same way.
    if isinstance(indexer, da.Array):
        indexer = indexer.rechunk((axis_old_chunks,))
    else:
        indexer = da.from_array(indexer, chunks=(axis_old_chunks,))

    # Apply the indexing operation.
    v = da.compress(indexer, data, axis=axis)

    # Need to compute chunks sizes in order to know dimension sizes;
    # would normally do v.compute_chunk_sizes() but that is slow for
    # multidimensional arrays, so hack something more efficient.
    axis_new_chunks_list = []
    slice_start = 0
    need_rechunk = False
    for old_chunk_size in axis_old_chunks:
        slice_stop = slice_start + old_chunk_size
        new_chunk_size = int(np.sum(indexer_computed[slice_start:slice_stop]))
        if new_chunk_size == 0:
            need_rechunk = True
        axis_new_chunks_list.append(new_chunk_size)
        slice_start = slice_stop
    axis_new_chunks = tuple(axis_new_chunks_list)
    new_chunks = tuple(
        [axis_new_chunks if i == axis else c for i, c in enumerate(old_chunks)]
    )
    v._chunks = new_chunks

    # Deal with empty chunks, they break reductions.
    # Possibly related to https://github.com/dask/dask/issues/10327
    # and https://github.com/dask/dask/issues/2794
    if need_rechunk:
        axis_new_chunks_nonzero = tuple([x for x in axis_new_chunks if x > 0])
        # Edge case, all chunks empty:
        if len(axis_new_chunks_nonzero) == 0:
            # Not much we can do about this, no data.
            axis_new_chunks_nonzero = (0,)
        new_chunks_nonzero = tuple(
            [
                axis_new_chunks_nonzero if i == axis else c
                for i, c in enumerate(new_chunks)
            ]
        )
        v = v.rechunk(new_chunks_nonzero)

    return v


def init_filesystem(url, **kwargs):
    """Initialise a fsspec filesystem from a given base URL and parameters."""

    # Special case Google Cloud Storage, authenticate the user.
    if "gs://" in url or "gcs://" in url:
        if colab is not None:  # pragma: no cover
            # We are in colab, use colab's built-in authentication function.
            colab.auth.authenticate_user()
        else:
            # Assume user has performed gcloud auth application-default login
            pass
        import google.auth  # type: ignore

        # Load application-default credentials.
        with warnings.catch_warnings():
            # Warnings are generally not that useful here, silence them.
            warnings.simplefilter("ignore")

            # To make this work with a service account on github actions, the
            # scopes parameter is needed, see also:
            # https://stackoverflow.com/a/74562563/761177
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        kwargs.setdefault("token", credentials)

        # Ensure options are passed through to gcsfs, even if URL is chained.
        if url.startswith("gs://") or url.startswith("gcs://"):
            storage_options = kwargs
        elif "gs://" in url:
            # Chained URL.
            storage_options = {"gs": kwargs}
        elif "gcs://" in url:
            # Chained URL.
            storage_options = {"gcs": kwargs}

    elif "s3://" in url:
        # N.B., we currently assume any S3 URLs refer to buckets hosted at Sanger.
        config = {
            "signature_version": "s3",
            "s3": {"addressing_style": "virtual"},
        }

        # Create an S3FileSystem with custom endpoint if specified.
        kwargs.setdefault("anon", True)  # Default to anonymous access.
        kwargs.setdefault("endpoint_url", "https://cog.sanger.ac.uk")
        kwargs.setdefault("config_kwargs", config)

        if url.startswith("s3://"):
            storage_options = kwargs
        else:
            # Chained URL.
            storage_options = {"s3": kwargs}

    else:
        # Some other kind of URL, pass through kwargs as-is.
        storage_options = kwargs

    # Process the URL using fsspec.
    fs, path = url_to_fs(url, **storage_options)

    # Path compatibility, fsspec/gcsfs behaviour varies between versions.
    while path.endswith("/"):
        path = path[:-1]

    return fs, path


def init_zarr_store(fs, path):
    """Initialise a zarr store (mapping) from a fsspec filesystem."""

    return SafeStore(FSMap(fs=fs, root=path, check=False, create=False))


# N.B., previously Region was defined as a named tuple. However, this led to
# some subtle bugs where instances where treated as normal tuples. So to avoid
# confusion, create a dedicated class.


class Region:
    """A region of a reference genome, i.e., a contig or contig interval."""

    def __init__(self, contig, start=None, end=None):
        self._contig = contig
        self._start = start
        self._end = end

    @property
    def contig(self):
        return self._contig

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def __hash__(self):
        return hash((self.contig, self.start, self.end))

    def __eq__(self, other):
        return (
            isinstance(other, Region)
            and (self.contig == other.contig)
            and (self.start == other.start)
            and (self.end == other.end)
        )

    def __str__(self):
        out = self._contig
        if self._start is not None or self._end is not None:
            out += ":"
            if self._start is not None:
                out += f"{self._start:,}"
            out += "-"
            if self.end is not None:
                out += f"{self._end:,}"
        return out

    def to_dict(self):
        return dict(
            contig=self.contig,
            start=self.start,
            end=self.end,
        )


def _handle_region_coords(resource, region):
    region_pattern_match = re.search("([a-zA-Z0-9_]+):(.+)-(.+)", region)
    if region_pattern_match:
        # parse region string that contains genomic coordinates
        region_split = region_pattern_match.groups()
        contig = region_split[0]
        start = int(region_split[1].replace(",", ""))
        end = int(region_split[2].replace(",", ""))

        if contig not in _valid_contigs(resource):
            raise ValueError(
                f"The genomic region {region!r} is invalid because contig {contig!r} does not exist in the dataset."
            )
        else:
            contig_length = resource.genome_sequence(region=contig).shape[0]
            if start < 1 or end < start or end > contig_length:
                raise ValueError(
                    f"The genomic region {region!r} is invalid for contig {contig!r} with length {contig_length}."
                )

        return Region(contig, start, end)

    else:
        return None


def _prep_geneset_attributes_arg(attributes):
    if type(attributes) not in [tuple, list] and attributes != "*":
        raise TypeError("'attributes' must be a list, tuple, or '*'")
    if attributes is not None:
        attributes = tuple(attributes)
    return attributes


def _handle_region_feature(resource, region):
    if hasattr(resource, "genome_features"):
        gene_annotation = resource.genome_features(attributes=["ID"])
        results = gene_annotation.query(f"ID == '{region}'")
        if not results.empty:
            # the region is a feature ID
            feature = results.squeeze()
            return Region(feature.contig, int(feature.start), int(feature.end))


def _valid_contigs(resource):
    """Determine which contig identifiers are valid for the given data resource."""
    valid_contigs = tuple(resource.contigs)
    # allow for optional virtual contigs
    virtual_contigs = getattr(resource, "virtual_contigs", None)
    if virtual_contigs is not None:
        valid_contigs += tuple(virtual_contigs.keys())
    return valid_contigs


single_region_param_type: TypeAlias = Union[str, Region, Mapping]

region_param_type: TypeAlias = Union[
    single_region_param_type,
    List[single_region_param_type],
    Tuple[single_region_param_type, ...],
]

single_contig_param_type: TypeAlias = str

contig_param_type: TypeAlias = Union[
    single_contig_param_type,
    List[single_contig_param_type],
    Tuple[single_contig_param_type, ...],
]


def parse_single_region(resource, region: single_region_param_type) -> Region:
    if isinstance(region, Region):
        # The region is already a Region, nothing to do.
        return region

    if isinstance(region, Mapping):
        # The region is in dictionary form, convert to Region instance.
        return Region(
            contig=region.get("contig"),
            start=region.get("start"),
            end=region.get("end"),
        )

    assert isinstance(region, str)

    # check if region is a whole contig
    if region in _valid_contigs(resource):
        return Region(region, None, None)

    # check if region is a region string providing coordinates
    region_from_coords = _handle_region_coords(resource, region)
    if region_from_coords is not None:
        return region_from_coords

    # check if region is a gene annotation feature ID
    region_from_feature = _handle_region_feature(resource, region)
    if region_from_feature is not None:
        return region_from_feature

    raise ValueError(
        f"Region {region!r} is not a valid contig, region string or feature ID."
    )


def parse_multi_region(
    resource,
    region: region_param_type,
) -> List[Region]:
    if isinstance(region, (list, tuple)):
        return [parse_single_region(resource, r) for r in region]
    else:
        return [parse_single_region(resource, region)]


def resolve_region(
    resource,
    region: region_param_type,
) -> Union[Region, List[Region]]:
    """Parse the provided region and return a `Region` object or list of
    `Region` objects if multiple values provided.

    Supports contig names, gene names and genomic coordinates.

    """
    if isinstance(region, (list, tuple)):
        # Multiple regions, normalise to list and resolve components.
        return [parse_single_region(resource, r) for r in region]
    else:
        return parse_single_region(resource, region)


def locate_region(region: Region, pos: np.ndarray) -> slice:
    """Get array slice and a parsed genomic region.

    Parameters
    ----------
    region : Region
        The region to locate.
    pos : array-like
        Positions to be searched.

    Returns
    -------
    loc_region : slice

    """
    pos_idx = allel.SortedIndex(pos)
    try:
        loc_region = pos_idx.locate_range(region.start, region.end)
    except KeyError:
        # There are no data within the requested region, return a zero-length slice.
        loc_region = slice(0, 0)
    return loc_region


def region_str(region: List[Region]) -> str:
    """Convert a region to a string representation.

    Parameters
    ----------
    region : Region or list of Region
        The region to display.

    Returns
    -------
    out : str

    """
    if isinstance(region, list):
        if len(region) > 1:
            return "; ".join([str(r) for r in region])
        else:
            return str(region[0])
    else:
        return str(region)


def _simple_xarray_concat_arrays(
    datasets: List[xr.Dataset], names: List[Hashable], dim: str
) -> Mapping[Hashable, xr.DataArray]:
    # Access the first dataset, this will be used as the template for
    # any arrays that don't need to be concatenated.
    ds0 = datasets[0]

    # Set up return value, collection of concatenated arrays.
    out: Dict[Hashable, xr.DataArray] = dict()

    # Iterate over variable names.
    for k in names:
        # Access the variable from the first dataset.
        v = ds0[k]

        if dim in v.dims:
            # Dimension to be concatenated is present, need to concatenate.

            # Figure out which axis corresponds to the given dimension.
            axis = v.dims.index(dim)

            # Access the xarray DataArrays to be concatenated.
            xr_arrays = [ds[k] for ds in datasets]

            # Check that all arrays have the same dimension as the same axis.
            assert all([a.dims[axis] == dim for a in xr_arrays])

            # Access the inner arrays - these are either numpy or dask arrays.
            inner_arrays = [a.data for a in xr_arrays]

            # Concatenate inner arrays, depending on their type.
            if isinstance(inner_arrays[0], da.Array):
                concatenated_array = da.concatenate(inner_arrays, axis=axis)
            else:
                concatenated_array = np.concatenate(inner_arrays, axis=axis)

            # Store the result.
            out[k] = xr.DataArray(data=concatenated_array, dims=v.dims)

        else:
            # No concatenation is needed, keep the variable from the first dataset.
            out[k] = v

    return out


def simple_xarray_concat(
    datasets: List[xr.Dataset], dim: str, attrs: Optional[Mapping] = None
) -> xr.Dataset:
    # Access the first dataset, this will be used as the template for
    # any arrays that don't need to be concatenated.
    ds0 = datasets[0]

    if attrs is None:
        # Copy attributes from the first dataset.
        attrs = ds0.attrs

    if len(datasets) == 1:
        # Fast path, nothing to concatenate.
        return ds0

    # Concatenate coordinate variables.
    coords = _simple_xarray_concat_arrays(
        datasets=datasets,
        names=list(ds0.coords),
        dim=dim,
    )

    # Concatenate data variables.
    data_vars = _simple_xarray_concat_arrays(
        datasets=datasets,
        names=list(ds0.data_vars),
        dim=dim,
    )

    return xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)


# xarray concat() function is very slow, don't use for now
#
# def xarray_concat(
#     datasets,
#     dim,
#     data_vars="minimal",
#     coords="minimal",
#     compat="override",
#     join="override",
#     **kwargs,
# ):
#     if len(datasets) == 1:
#         return datasets[0]
#     else:
#         return xr.concat(
#             datasets,
#             dim=dim,
#             data_vars=data_vars,
#             coords=coords,
#             compat=compat,
#             join=join,
#             **kwargs,
#         )


def da_concat(arrays: List[da.Array], **kwargs) -> da.Array:
    if len(arrays) == 1:
        return arrays[0]
    else:
        return da.concatenate(arrays, **kwargs)


def value_error(
    name,
    value,
    expectation,
):
    message = (
        f"Bad value for parameter {name}; expected {expectation}, " f"found {value!r}"
    )
    raise ValueError(message)


def hash_params(params):
    """Helper function to hash function parameters."""
    s = json.dumps(params, sort_keys=True, indent=4)
    h = hashlib.md5(s.encode()).hexdigest()
    return h, s


def jitter(a, fraction):
    """Jitter data in `a` using the fraction `f`."""
    r = a.max() - a.min()
    return a + fraction * np.random.uniform(-r, r, a.shape)


class CacheMiss(Exception):
    pass


class LoggingHelper:
    def __init__(
        self, *, name: str, out: Optional[Union[str, IO]], debug: bool = False
    ):
        # set up a logger
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self._logger = logger

        # set up handler
        handler: Optional[logging.StreamHandler] = None
        if hasattr(out, "write"):
            handler = logging.StreamHandler(out)
        elif isinstance(out, str):
            handler = logging.FileHandler(out)
        self._handler = handler

        # configure handler
        if handler is not None:
            if debug:
                handler.setLevel(logging.DEBUG)
            else:
                handler.setLevel(logging.INFO)
            formatter = logging.Formatter(fmt="[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def flush(self):
        if self._handler is not None:
            self._handler.flush()

    def debug(self, msg):
        # get the name of the calling function, helps with debugging
        caller_name = sys._getframe().f_back.f_code.co_name
        msg = f"{caller_name}: {msg}"
        self._logger.debug(msg)

        # flush messages immediately
        self.flush()

    def info(self, msg):
        self._logger.info(msg)

        # flush messages immediately
        self.flush()

    def set_level(self, level):
        if self._handler is not None:
            self._handler.setLevel(level)


def jackknife_ci(stat_data, jack_stat, confidence_level):
    """Compute a confidence interval from jackknife resampling.

    Parameters
    ----------
    stat_data : scalar
        Value of the statistic computed on all data.
    jack_stat : ndarray
        Values of the statistic computed for each jackknife resample.
    confidence_level : float
        Desired confidence level (e.g., 0.95).

    Returns
    -------
    estimate
        Bias-corrected "jackknifed estimate".
    bias
        Jackknife bias.
    std_err
        Standard error.
    ci_err
        Size of the confidence interval.
    ci_low
        Lower limit of confidence interval.
    ci_upp
        Upper limit of confidence interval.

    Notes
    -----
    N.B., this implementation is based on code from astropy, see:

    https://github.com/astropy/astropy/blob/8aba9632597e6bb489488109222bf2feff5835a6/astropy/stats/jackknife.py#L55

    """
    from scipy.special import erfinv  # type: ignore

    n = len(jack_stat)

    mean_jack_stat = np.mean(jack_stat)

    # jackknife bias
    bias = (n - 1) * (mean_jack_stat - stat_data)

    # jackknife standard error
    std_err = np.sqrt(
        (n - 1) * np.mean((jack_stat - mean_jack_stat) * (jack_stat - mean_jack_stat))
    )

    # bias-corrected "jackknifed estimate"
    estimate = stat_data - bias

    # confidence interval
    z_score = np.sqrt(2.0) * erfinv(confidence_level)
    ci_err = 2 * z_score * std_err
    ci_low, ci_upp = estimate + z_score * np.array((-std_err, std_err))

    return estimate, bias, std_err, ci_err, ci_low, ci_upp


def plotly_discrete_legend(
    color,
    color_values,
    **kwargs,
):
    """Manually create a legend by making a scatter plot then
    hiding everything but the legend.

    Parameters
    ----------
    color : str
        Name of field used to obtain categorical values.
    color_values : list
        Categorical values to map to colours.
    **kwargs
        Passed through to px.scatter().

    Returns
    -------
    fig : Figure
        Plotly figure.

    """

    data_frame = pd.DataFrame(
        {
            color: color_values,
            "x": np.zeros(len(color_values)),
            "y": np.zeros(len(color_values)),
        }
    )

    fig = px.scatter(
        data_frame=data_frame,
        x="x",
        y="y",
        color=color,
        template="simple_white",
        range_x=[1, 2],  # hide the scatter points
        range_y=[1, 2],
        **kwargs,
    )

    # visual styling to hide everything but the legend
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
            itemsizing="constant",
            orientation="v",
            title=color.capitalize(),
        ),
        xaxis=dict(
            visible=False,
        ),
        yaxis=dict(
            visible=False,
        ),
        margin=dict(
            autoexpand=False,
            pad=0,
            t=0,
            r=0,
            b=0,
            l=0,
        ),
    )

    return fig


def get_gcp_region(details):
    """Attempt to determine the current GCP region based on
    response from ipinfo."""

    if details is not None:
        org = details.org
        country = details.country
        region = details.region
        if org == "AS396982 Google LLC":
            if country == "US":
                if region == "Iowa":
                    return "us-central1"
                elif region == "South Carolina":
                    return "us-east1"
                elif region == "Virginia":
                    return "us-east4"
                elif region == "Ohio":
                    return "us-east5"
                elif region == "Oregon":
                    return "us-west1"
                elif region == "California":
                    return "us-west2"
                elif region == "Utah":
                    return "us-west3"
                elif region == "Nevada":
                    return "us-west4"
                elif region == "Texas":
                    return "us-south1"
                else:
                    return "us-other"
            elif country == "ZA":
                if region == "Gauteng":
                    return "africa-south1"
            # Add other regions later if needed.
            return "other"
    return None


class ColabLocationError(RuntimeError):
    pass


def check_colab_location(gcp_region: Optional[str]):
    """
    Sometimes, colab will allocate a VM outside the US, e.g., in
    Europe or Asia. Because the MalariaGEN GCS buckets are located
    in the US, this is usually bad for performance, because of
    increased latency and lower bandwidth, and costs money. Add a
    check for this and raise an error if not in the US.
    """

    if colab and gcp_region:
        if not gcp_region.startswith("us-"):
            raise ColabLocationError(
                fill(
                    dedent(
                        """
                Your Google Colab VM is not located in the US.
                Please select "Runtime > Disconnect and delete runtime" from
                the menu to request a new VM, then rerun your notebook.
            """
                    )
                )
            )


def check_types(f):
    """Simple decorator to provide runtime checking of parameter types.

    N.B., the typeguard package does have a decorator function called
    @typechecked which performs a similar purpose. However, the typeguard
    decorator causes a memory leak and doesn't seem usable. Also, the
    typeguard decorator performs runtime checking of all variables within
    the function as well as the arguments and return values. We only want
    checking of the arguments to help users provide correct inputs.

    """

    @wraps(f)
    def check_types_wrapper(*args, **kwargs):
        type_hints = get_type_hints(f)
        call_args = getcallargs(f, *args, **kwargs)
        for k, t in type_hints.items():
            if k in call_args:
                v = call_args[k]
                try:
                    typeguard.check_type(v, t)
                except typeguard.TypeCheckError as e:
                    expected_type = humanize_type(t)
                    actual_type = humanize_type(type(v))
                    message = fill(
                        dedent(
                            f"""
                        Parameter {k!r} with value {v!r} in call to function {f.__name__!r} has incorrect type:
                        found {actual_type}, expected {expected_type}. See below for further information.
                    """
                        )
                    )
                    message += f"\n\n{e}"
                    error = TypeError(message)
                    raise error from None
        return f(*args, **kwargs)

    return check_types_wrapper


@numba.njit
def true_runs(a):
    in_run = False
    starts = []
    stops = []
    for i in range(a.shape[0]):
        v = a[i]
        if not in_run and v:
            in_run = True
            starts.append(i)
        if in_run and not v:
            in_run = False
            stops.append(i)
    if in_run:
        stops.append(a.shape[0])
    return np.array(starts, dtype=np.int64), np.array(stops, dtype=np.int64)


@numba.njit(parallel=True)
def pdist_abs_hamming(X):
    n_obs = X.shape[0]
    n_ftr = X.shape[1]
    out = np.zeros((n_obs, n_obs), dtype=np.int32)
    for i in range(n_obs):
        x = X[i]
        for j in numba.prange(i + 1, n_obs):
            y = X[j]
            d = 0
            for k in range(n_ftr):
                if x[k] != y[k]:
                    d += 1
            out[i, j] = d
            out[j, i] = d
    return out


@numba.njit
def square_to_condensed(i, j, n):
    """Convert distance matrix coordinates from square form (i, j) to condensed form."""

    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n * j - j * (j + 1) // 2 + i - 1 - j


@numba.njit(parallel=True)
def multiallelic_diplotype_pdist(X, metric):
    """Optimised implementation of pairwise distance between diplotypes.

    N.B., here we assume the array X provides diplotypes as genotype allele
    counts, with axes in the order (n_samples, n_sites, n_alleles).

    Computation will be faster if X is a contiguous (C order) array.

    The metric argument is the function to compute distance for a pair of
    diplotypes. This can be a numba jitted function.

    """
    n_samples = X.shape[0]
    n_pairs = (n_samples * (n_samples - 1)) // 2
    out = np.zeros(n_pairs, dtype=np.float32)

    # Loop over samples, first in pair.
    for i in range(n_samples):
        x = X[i, :, :]

        # Loop over observations again, second in pair.
        for j in numba.prange(i + 1, n_samples):
            y = X[j, :, :]

            # Compute distance for the current pair.
            d = metric(x, y)

            # Store result for the current pair.
            k = square_to_condensed(i, j, n_samples)
            out[k] = d

    return out


@numba.njit
def multiallelic_diplotype_mean_cityblock(x, y):
    """Compute the mean cityblock distance between two diplotypes x and y. The
    diplotype vectors are expected as genotype allele counts, i.e., x and y
    should have the same shape (n_sites, n_alleles).

    N.B., here we compute the mean value of the distance over sites where
    both individuals have a called genotype. This avoids computing distance
    at missing sites.

    """
    n_sites = x.shape[0]
    n_alleles = x.shape[1]
    distance = np.float32(0)
    n_sites_called = np.float32(0)

    # Loop over sites.
    for i in range(n_sites):
        x_is_called = False
        y_is_called = False
        d = np.float32(0)

        # Loop over alleles.
        for j in range(n_alleles):
            # Access allele counts.
            xc = np.float32(x[i, j])
            yc = np.float32(y[i, j])

            # Check if any alleles observed.
            x_is_called = x_is_called or (xc > 0)
            y_is_called = y_is_called or (yc > 0)

            # Compute cityblock distance (absolute difference).
            d += np.fabs(xc - yc)

        # Accumulate distance for the current pair, but only if both samples
        # have a called genotype.
        if x_is_called and y_is_called:
            distance += d
            n_sites_called += np.float32(1)

    # Compute the mean distance over sites with called genotypes.
    if n_sites_called > 0:
        mean_distance = distance / n_sites_called
    else:
        mean_distance = np.nan

    return mean_distance


@numba.njit
def multiallelic_diplotype_sqeuclidean(x, y):
    n_sites = x.shape[0]
    n_alleles = x.shape[1]
    distance = np.float32(0)
    n_sites_called = np.float32(0)

    # Loop over sites.
    for i in range(n_sites):
        x_is_called = False
        y_is_called = False
        d = np.float32(0)

        # Loop over alleles.
        for j in range(n_alleles):
            # Access allele counts.
            xc = np.float32(x[i, j])
            yc = np.float32(y[i, j])

            # Check if any alleles observed.
            x_is_called = x_is_called or (xc > 0)
            y_is_called = y_is_called or (yc > 0)

            # Compute squared euclidean distance.
            d += (xc - yc) ** 2

        # Accumulate distance for the current pair, but only if both samples
        # have a called genotype.
        if x_is_called and y_is_called:
            distance += d
            n_sites_called += np.float32(1)

    return distance, n_sites_called


@numba.njit
def multiallelic_diplotype_mean_sqeuclidean(x, y):
    """Compute the mean squared euclidean distance between two diplotypes x and
    y. The diplotype vectors are expected as genotype allele counts, i.e., x and
    y should have the same shape (n_sites, n_alleles).

    N.B., here we compute the mean value of the distance over sites where
    both individuals have a called genotype. This avoids computing distance
    at missing sites.

    """

    distance, n_sites_called = multiallelic_diplotype_sqeuclidean(x, y)

    # Compute the mean distance over sites with called genotypes.
    if n_sites_called > 0:
        mean_distance = distance / n_sites_called
    else:
        mean_distance = np.nan

    return mean_distance


@numba.njit
def multiallelic_diplotype_mean_euclidean(x, y):
    """Compute the mean euclidean distance between two diplotypes x and
    y. The diplotype vectors are expected as genotype allele counts, i.e., x and
    y should have the same shape (n_sites, n_alleles).

    N.B., here we compute the mean value of the distance over sites where
    both individuals have a called genotype. This avoids computing distance
    at missing sites.

    """

    sqdistance, n_sites_called = multiallelic_diplotype_sqeuclidean(x, y)
    distance = np.sqrt(sqdistance)

    # Compute the mean distance over sites with called genotypes.
    if n_sites_called > 0:
        mean_distance = distance / n_sites_called
    else:
        mean_distance = np.nan

    return mean_distance


@numba.njit
def trim_alleles(ac):
    """Remap allele indices to trim out unobserved alleles.

    The idea here is that our SNP data includes alleles which
    may not ever be observed, or may not be observed within a
    particular subset of the data. For convenience, it is useful
    to retain only alleles which are observed.

    Here we use a given set of allele counts (ac) to identify
    observed alleles and derive a mapping from old allele
    indices to a new set of allele indices.

    For example, if alleles 0 and 2 are observed at a given
    site in ac, then 0 will be mapped to 0 and 2 will be mapped
    to 1.

    Similarly, if alleles 1 and 3 are observed at a given site
    in ac, then 1 will be mapped to 0 and 3 will be mapped to 1.

    """

    n_variants = ac.shape[0]
    n_alleles = ac.shape[1]

    # Create the output array - this is an allele mapping array,
    # that specifies how to recode allele indices.
    mapping = np.empty((n_variants, n_alleles), dtype=np.int32)

    # The value of -1 indicates that there is no mapping for
    # a given allele. We fill with -1 so we can then set
    # values where we do observe alleles and can define a
    # mapping.
    mapping[:] = -1

    # Iterate over variants.
    for i in range(n_variants):
        # This will be the new index that we are mapping this allele to, if the
        # allele is observed.
        new_allele_index = 0

        # Iterate over columns (alleles) in the input array.
        for allele_index in range(n_alleles):
            # Access the count for the jth allele.
            c = ac[i, allele_index]
            if c > 0:
                # We have found a non-zero allele count, remap the allele.
                mapping[i, allele_index] = new_allele_index

                # Increment to be ready for the next allele.
                new_allele_index += 1

    return mapping


@numba.njit
def apply_allele_mapping(x, mapping, max_allele):
    """Transform an array x, where the columns correspond to alleles,
    according to an allele mapping.

    Values from columns in x will be move to different columns in the
    output according to the allele index mapping given.

    """
    n_sites = x.shape[0]
    n_alleles = x.shape[1]
    assert mapping.shape[0] == n_sites
    assert mapping.shape[1] == n_alleles

    # Create output array.
    out = np.empty(shape=(n_sites, max_allele + 1), dtype=x.dtype)

    # Iterate over sites.
    for i in range(n_sites):
        # Iterate over alleles.
        for allele_index in range(n_alleles):
            # Find the new index for this allele.
            new_allele_index = mapping[i, allele_index]

            # Copy data to the new allele index.
            if new_allele_index >= 0 and new_allele_index <= max_allele:
                out[i, new_allele_index] = x[i, allele_index]

    return out


def dask_apply_allele_mapping(v, mapping, max_allele):
    assert isinstance(v, da.Array)
    assert isinstance(mapping, np.ndarray)
    assert v.ndim == 2
    assert mapping.ndim == 2
    assert v.shape[0] == mapping.shape[0]
    v = v.rechunk((v.chunks[0], -1))
    mapping = da.from_array(mapping, chunks=(v.chunks[0], -1))
    out = da.map_blocks(
        lambda xb, mb: apply_allele_mapping(xb, mb, max_allele=max_allele),
        v,
        mapping,
        dtype=v.dtype,
        chunks=(v.chunks[0], [max_allele + 1]),
    )
    return out


def genotype_array_map_alleles(gt, mapping):
    # Transform genotype calls via an allele mapping.
    # N.B., scikit-allel does not handle empty blocks well, so we
    # include some extra logic to handle that better.
    assert isinstance(gt, np.ndarray)
    assert isinstance(mapping, np.ndarray)
    assert gt.ndim == 3
    assert mapping.ndim == 3
    assert gt.shape[0] == mapping.shape[0]
    assert gt.shape[1] > 0
    assert gt.shape[2] == 2
    if gt.size > 0:
        # Block is not empty, can pass through to GenotypeArray.
        assert gt.shape[0] > 0
        m = mapping[:, 0, :]
        out = allel.GenotypeArray(gt).map_alleles(m).values
    else:
        # Block is empty so no alleles need to be mapped.
        assert gt.shape[0] == 0
        out = gt
    return out


def dask_genotype_array_map_alleles(gt, mapping):
    assert isinstance(gt, da.Array)
    assert isinstance(mapping, np.ndarray)
    assert gt.ndim == 3
    assert mapping.ndim == 2
    assert gt.shape[0] == mapping.shape[0]
    mapping = da.from_array(mapping, chunks=(gt.chunks[0], -1))
    gt_out = da.map_blocks(
        genotype_array_map_alleles,
        gt,
        mapping[:, None, :],
        chunks=gt.chunks,
        dtype=gt.dtype,
    )
    return gt_out


def pandas_apply(f, df, columns):
    """Optimised alternative to pandas apply."""
    df = df.reset_index(drop=True)
    iterator = zip(*[df[c].values for c in columns])
    ret = pd.Series((f(*vals) for vals in iterator))
    return ret


def compare_series_like(actual, expect):
    """Compare pandas series-like objects for equality or floating point
    similarity, handling missing values appropriately."""

    # Handle object arrays, these don't get nans compared properly.
    t = actual.dtype
    if t == object:
        expect = expect.fillna("NA")
        actual = actual.fillna("NA")

    if t.kind == "f":
        assert_allclose(actual.values, expect.values)
    else:
        assert_array_equal(actual.values, expect.values)


@numba.njit
def hash_columns(x):
    # Here we want to compute a hash for each column in the
    # input array. However, we assume the input array is in
    # C contiguous order, and therefore we scan the array
    # and perform the computation in this order for more
    # efficient memory access.
    #
    # This function uses the DJBX33A hash function which
    # is much faster than computing Python hashes of
    # bytes, as discovered by Tom White in work on sgkit.
    m = x.shape[0]
    n = x.shape[1]
    out = np.empty(n, dtype=np.int64)
    out[:] = 5381
    for i in range(m):
        for j in range(n):
            v = x[i, j]
            out[j] = out[j] * 33 + v
    return out


def haplotype_frequencies(h):
    """Compute haplotype frequencies, returning a dictionary that maps
    haplotype hash values to frequencies."""
    n = h.shape[1]
    hashes = hash_columns(np.asarray(h))
    count = Counter(hashes)
    freqs = {key: count / n for key, count in count.items()}
    counts = {key: count for key, count in count.items()}
    nobs = {key: n for key, count in count.items()}
    return freqs, counts, nobs


def distributed_client():
    from distributed import get_client

    try:
        client = get_client()
    except ValueError:
        client = None
    return client


def _karyotype_tags_n_alt(gt, alts, inversion_alts):
    # could be Numba'd for speed but was already quick (not many inversion tag snps)
    n_sites = gt.shape[0]
    n_samples = gt.shape[1]

    # create empty array
    inv_n_alt = np.empty((n_sites, n_samples), dtype=np.int8)

    # for every site
    for i in range(n_sites):
        # find the index of the correct tag snp allele
        tagsnp_index = np.where(alts[i] == inversion_alts[i])[0]

        for j in range(n_samples):
            # count alleles which == tag snp allele and store
            n_tag_alleles = np.sum(gt[i, j] == tagsnp_index[0])
            inv_n_alt[i, j] = n_tag_alleles

    return inv_n_alt


def prep_samples_for_cohort_grouping(*, df_samples, area_by, period_by):
    # Take a copy, as we will modify the dataframe.
    df_samples = df_samples.copy()

    # Fix "intermediate" or "unassigned" taxon values - we only want to build
    # cohorts with clean taxon calls, so we set other values to None.
    loc_intermediate_taxon = (
        df_samples["taxon"].str.startswith("intermediate").fillna(False)
    )
    df_samples.loc[loc_intermediate_taxon, "taxon"] = None
    loc_unassigned_taxon = (
        df_samples["taxon"].str.startswith("unassigned").fillna(False)
    )
    df_samples.loc[loc_unassigned_taxon, "taxon"] = None

    # Add period column.
    if period_by == "year":
        make_period = _make_sample_period_year
    elif period_by == "quarter":
        make_period = _make_sample_period_quarter
    elif period_by == "month":
        make_period = _make_sample_period_month
    else:  # pragma: no cover
        raise ValueError(
            f"Value for period_by parameter must be one of 'year', 'quarter', 'month'; found {period_by!r}."
        )
    sample_period = df_samples.apply(make_period, axis="columns")
    df_samples["period"] = sample_period

    # Add area column for consistent output.
    df_samples["area"] = df_samples[area_by]

    return df_samples


def build_cohorts_from_sample_grouping(*, group_samples_by_cohort, min_cohort_size):
    # Build cohorts dataframe.
    df_cohorts = group_samples_by_cohort.agg(
        size=("sample_id", len),
        lat_mean=("latitude", "mean"),
        lat_max=("latitude", "max"),
        lat_min=("latitude", "min"),
        lon_mean=("longitude", "mean"),
        lon_max=("longitude", "max"),
        lon_min=("longitude", "min"),
    )
    # Reset index so that the index fields are included as columns.
    df_cohorts = df_cohorts.reset_index()

    # Add cohort helper variables.
    cohort_period_start = df_cohorts["period"].apply(lambda v: v.start_time)
    cohort_period_end = df_cohorts["period"].apply(lambda v: v.end_time)
    df_cohorts["period_start"] = cohort_period_start
    df_cohorts["period_end"] = cohort_period_end
    # Create a label that is similar to the cohort metadata,
    # although this won't be perfect.
    df_cohorts["label"] = df_cohorts.apply(
        lambda v: f"{v.area}_{v.taxon[:4]}_{v.period}", axis="columns"
    )

    # Apply minimum cohort size.
    df_cohorts = df_cohorts.query(f"size >= {min_cohort_size}").reset_index(drop=True)

    # Early check for no cohorts.
    if len(df_cohorts) == 0:
        raise ValueError(
            "No cohorts available for the given sample selection parameters and minimum cohort size."
        )

    return df_cohorts


def add_frequency_ci(*, ds, ci_method):
    from statsmodels.stats.proportion import proportion_confint  # type: ignore

    if ci_method is not None:
        count = ds["event_count"].values
        nobs = ds["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frq_ci_low, frq_ci_upp = proportion_confint(
                count=count, nobs=nobs, method=ci_method
            )
        ds["event_frequency_ci_low"] = ("variants", "cohorts"), frq_ci_low
        ds["event_frequency_ci_upp"] = ("variants", "cohorts"), frq_ci_upp


def _make_sample_period_month(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="M", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_quarter(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="Q", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_year(row):
    year = row.year
    if year > 0:
        return pd.Period(freq="Y", year=year)
    else:
        return pd.NaT
