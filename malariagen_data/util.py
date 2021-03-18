from enum import Enum
from urllib.parse import unquote_plus
import pandas
import numpy as np
from collections.abc import Mapping
import dask.array as da


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
    "seqid",
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


def from_zarr(z, inline_array):
    """Utility function for turning a zarr array into a dask array.

    N.B., dask does have it's own from_zarr() function but we roll
    our own here to get a little more control.

    """
    kwargs = dict(chunks=z.chunks, fancy=False, lock=False, inline_array=inline_array)
    try:
        d = da.from_array(z, **kwargs)
    except TypeError:
        # only later versions of dask support inline_array argument
        del kwargs["inline_array"]
        d = da.from_array(z, **kwargs)
    return d
