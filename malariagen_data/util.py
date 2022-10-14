import hashlib
import json
import logging
import re
import sys
from collections.abc import Mapping
from enum import Enum
from urllib.parse import unquote_plus

import allel
import dask.array as da
import numpy as np
import pandas
import pandas as pd
import plotly.express as px
import xarray as xr
from fsspec.core import url_to_fs
from fsspec.mapping import FSMap

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


def unpack_gff3_attributes(df, attributes):

    df = df.copy()

    # discover all attribute keys
    all_attributes = set()
    for a in df["attributes"]:
        all_attributes.update(a.keys())
    all_attributes = sorted(all_attributes)

    if attributes == tuple("*"):
        attributes = all_attributes

    # unpack attributes into columns
    for key in attributes:
        if key not in all_attributes:
            raise ValueError(f"'{key}' not in attributes set. Options {all_attributes}")
        df[key] = df["attributes"].apply(lambda v: v.get(key, np.nan))
    del df["attributes"]

    return df


# zarr compatibility, version 2.11.0 introduced the BaseStore class
# see also https://github.com/malariagen/malariagen-data-python/issues/129

try:
    # zarr >= 2.11.0
    # noinspection PyUnresolvedReferences
    from zarr.storage import KVStore

    class SafeStore(KVStore):
        def __getitem__(self, key):
            try:
                return self._mutable_mapping[key]
            except KeyError as e:
                # raise a different error to ensure zarr propagates the exception, rather than filling
                raise FileNotFoundError(e)

        def __contains__(self, key):
            return key in self._mutable_mapping

except ImportError:
    # zarr < 2.11.0

    class SafeStore(Mapping):
        def __init__(self, store):
            self.store = store

        def __getitem__(self, key):
            try:
                return self.store[key]
            except KeyError as e:
                # raise a different error to ensure zarr propagates the exception, rather than filling
                raise FileNotFoundError(e)

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


def da_from_zarr(z, inline_array, chunks="auto"):
    """Utility function for turning a zarr array into a dask array.

    N.B., dask does have its own from_zarr() function, but we roll
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
        v = da_compress(indexer, a.data, axis)

    return v


def da_compress(indexer, data, axis):
    """Wrapper for dask.array.compress() which computes chunk sizes faster."""

    # sanity checks
    assert isinstance(data, da.Array)
    assert isinstance(indexer, da.Array)
    assert isinstance(axis, int)
    assert indexer.shape[0] == data.shape[axis]

    # useful variables
    old_chunks = data.chunks
    axis_old_chunks = old_chunks[axis]
    axis_n_chunks = len(axis_old_chunks)

    # ensure indexer and data are chunked in the same way
    indexer = indexer.rechunk((axis_old_chunks,))

    # apply the indexing operation
    v = da.compress(indexer, data, axis=axis)

    # need to compute chunks sizes in order to know dimension sizes;
    # would normally do v.compute_chunk_sizes() but that is slow for
    # multidimensional arrays, so hack something more efficient

    axis_new_chunks = tuple(
        indexer.map_blocks(
            lambda b: np.sum(b, keepdims=True),
            chunks=((1,) * axis_n_chunks,),
        ).compute()
    )
    new_chunks = tuple(
        [axis_new_chunks if i == axis else c for i, c in enumerate(old_chunks)]
    )
    v._chunks = new_chunks

    return v


def init_filesystem(url, **kwargs):
    """Initialise a fsspec filesystem from a given base URL and parameters."""

    # special case Google Cloud Storage, use anonymous access, avoids a delay
    if url.startswith("gs://") or url.startswith("gcs://"):
        kwargs["token"] = "anon"
    elif "gs://" in url:
        # chained URL
        kwargs["gs"] = dict(token="anon")
    elif "gcs://" in url:
        # chained URL
        kwargs["gcs"] = dict(token="anon")

    # process the url using fsspec
    fs, path = url_to_fs(url, **kwargs)

    # path compatibility, fsspec/gcsfs behaviour varies between version
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

        if contig not in resource.contigs:
            raise ValueError(f"Contig {contig} does not exist in the dataset.")
        elif (
            start < 0
            or end <= start
            or end > resource.genome_sequence(region=contig).shape[0]
        ):
            raise ValueError("Provided genomic coordinates are not valid.")

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
    gene_annotation = resource.genome_features(attributes=["ID"])
    results = gene_annotation.query(f"ID == '{region}'")
    if not results.empty:
        # region is a feature ID
        feature = results.squeeze()
        return Region(feature.contig, int(feature.start), int(feature.end))
    else:
        return None


def resolve_region(resource, region):
    """Parse the provided region and return `Region(contig, start, end)`.
    Supports contig names, gene names and genomic coordinates"""

    if isinstance(region, Region):
        # region is already Region tuple, nothing to do
        return region

    if isinstance(region, dict):
        return Region(
            contig=region.get("contig"),
            start=region.get("start"),
            end=region.get("end"),
        )

    if isinstance(region, (list, tuple)):
        # multiple regions, normalise to list and resolve components
        return [resolve_region(resource, r) for r in region]

    # check type, fail early if bad
    if not isinstance(region, str):
        raise TypeError("The region parameter must be a string or Region object.")

    # check if region is a chromosome arm
    if region in resource.contigs:
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


def locate_region(region, pos):
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
    pos = allel.SortedIndex(pos)
    loc_region = pos.locate_range(region.start, region.end)
    return loc_region


def xarray_concat(
    datasets,
    dim,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="override",
    **kwargs,
):
    if len(datasets) == 1:
        return datasets[0]
    else:
        return xr.concat(
            datasets,
            dim=dim,
            data_vars=data_vars,
            coords=coords,
            compat=compat,
            join=join,
            **kwargs,
        )


def type_error(
    name,
    value,
    expectation,
):
    message = (
        f"Bad type for parameter {name}; expected {expectation}, "
        f"found {type(value)}"
    )
    raise TypeError(message)


def value_error(
    name,
    value,
    expectation,
):
    message = (
        f"Bad value for parameter {name}; expected {expectation}, " f"found {value!r}"
    )
    raise ValueError(message)


def check_type(
    name,
    value,
    expectation,
):
    if not isinstance(value, expectation):
        type_error(name, value, expectation)


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
    def __init__(self, name, out, debug=False):

        # set up a logger
        logger = logging.getLogger(name)
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self._logger = logger

        # set up handler
        handler = None
        if isinstance(out, str):
            handler = logging.FileHandler(out)
        elif hasattr(out, "write"):
            handler = logging.StreamHandler(out)
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
    from scipy.special import erfinv

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
