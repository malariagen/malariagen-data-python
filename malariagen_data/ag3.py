import json
import sys
import warnings
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from textwrap import dedent

import allel
import dask
import dask.array as da
import ipinfo
import numba
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm.auto import tqdm
from tqdm.dask import TqdmCallback

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

import malariagen_data

from . import veff
from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    CacheMiss,
    LoggingHelper,
    Region,
    da_compress,
    da_from_zarr,
    dask_compress_dataset,
    hash_params,
    init_filesystem,
    init_zarr_store,
    jackknife_ci,
    jitter,
    locate_region,
    read_gff3,
    resolve_region,
    type_error,
    unpack_gff3_attributes,
    xarray_concat,
)

# silence dask performance warnings
dask.config.set(**{"array.slicing.split_large_chunks": False})

GCS_URL = "gs://vo_agam_release/"

GENOME_FASTA_PATH = (
    "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.fa"
)
GENOME_FAI_PATH = (
    "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.fa.fai"
)
GENOME_ZARR_PATH = (
    "reference/genome/agamp4/Anopheles-gambiae-PEST_CHROMOSOMES_AgamP4.zarr"
)

CONTIGS = "2R", "2L", "3R", "3L", "X"
DEFAULT_GENOME_PLOT_WIDTH = 800  # width in px for bokeh genome plots
DEFAULT_GENES_TRACK_HEIGHT = 100  # height in px for bokeh genes track plots
DEFAULT_MAX_COVERAGE_VARIANCE = 0.2


AA_CHANGE_QUERY = (
    "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
)

# Note regarding release identifiers and storage paths. Within the
# data storage, we have used path segments like "v3", "v3.1", "v3.2",
# etc., to separate data from different releases. There is an inconsistency
# in this convention, because the "v3" should have been "v3.0". To
# make the API more consistent, we would like to use consistent release
# identifiers like "3.0", "3.1", "3.2", etc., as parameter values and
# when release identifiers are added to returned dataframes. In order to
# achieve this, below we define two functions that allow mapping between
# these consistent release identifiers, and the less consistent release
# storage path segments.


def _release_to_path(release):
    """Compatibility function, allows us to use release identifiers like "3.0"
    and "3.1" in the public API, and map these internally into storage path
    segments."""
    if release == "3.0":
        # special case
        return "v3"
    elif release.startswith("3."):
        return f"v{release}"
    else:
        raise ValueError(f"Invalid release: {release!r}")


def _path_to_release(path):
    """Compatibility function, allows us to use release identifiers like "3.0"
    and "3.1" in the public API, and map these internally into storage path
    segments."""
    if path == "v3":
        return "3.0"
    elif path.startswith("v3."):
        return path[1:]
    else:
        raise RuntimeError(f"Unexpected release path: {path!r}")


class Ag3:
    """Provides access to data from Ag3.x releases.

    Parameters
    ----------
    url : str
        Base path to data. Give "gs://vo_agam_release/" to use Google Cloud
        Storage, or a local path on your file system if data have been
        downloaded.
    cohorts_analysis : str
        Cohort analysis version.
    species_analysis : {"aim_20200422", "pca_20200422"}, optional
        Species analysis version.
    site_filters_analysis : str, optional
        Site filters analysis version.
    bokeh_output_notebook : bool, optional
        If True (default), configure bokeh to output plots to the notebook.
    results_cache : str, optional
        Path to directory on local file system to save results.
    log : str or stream, optional
        File path or stream output for logging messages.
    debug : bool, optional
        Set to True to enable debug level logging.
    show_progress : bool, optional
        If True, show a progress bar during longer-running computations.
    check_location : bool, optional
        If True, use ipinfo to check the location of the client system.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> ag3 = malariagen_data.Ag3()

    Access data downloaded to a local file system:

        >>> ag3 = malariagen_data.Ag3("/local/path/to/vo_agam_release/")

    Access data from Google Cloud Storage, with caching on the local file system
    in a directory named "gcs_cache":

        >>> ag3 = malariagen_data.Ag3(
        ...     "simplecache::gs://vo_agam_release",
        ...     simplecache=dict(cache_storage="gcs_cache"),
        ... )

    Set up caching of some longer-running computations on the local file system,
    in a directory named "results_cache":

        >>> ag3 = malariagen_data.Ag3(results_cache="results_cache")

    """

    contigs = CONTIGS

    def __init__(
        self,
        url=GCS_URL,
        bokeh_output_notebook=True,
        results_cache=None,
        log=sys.stdout,
        debug=False,
        show_progress=True,
        check_location=True,
        cohorts_analysis=None,
        species_analysis=None,
        site_filters_analysis=None,
        **kwargs,
    ):

        self._url = url
        self._pre = kwargs.pop("pre", False)

        self._debug = debug
        self._show_progress = show_progress

        # set up logging
        self._log = LoggingHelper(name=__name__, out=log, debug=debug)

        # set up filesystem
        self._fs, self._base_path = init_filesystem(url, **kwargs)

        # load config.json
        path = f"{self._base_path}/v3-config.json"
        with self._fs.open(path) as f:
            config = json.load(f)

        self._public_releases = tuple(config["PUBLIC_RELEASES"])
        self._genome_feature_GFF_path = config["GENESET_GFF3_PATH"]
        if cohorts_analysis is None:
            self._cohorts_analysis = config["DEFAULT_COHORTS_ANALYSIS"]
        else:
            self._cohorts_analysis = cohorts_analysis
        if species_analysis is None:
            self._species_analysis = config["DEFAULT_SPECIES_ANALYSIS"]
        else:
            self._species_analysis = species_analysis
        if site_filters_analysis is None:
            self._site_filters_analysis = config["DEFAULT_SITE_FILTERS_ANALYSIS"]
        else:
            self._site_filters_analysis = site_filters_analysis

        # set up caches
        self._cache_releases = None
        self._cache_sample_sets = dict()
        self._cache_sample_set_to_release = None
        self._cache_general_metadata = dict()
        self._cache_species_calls = dict()
        self._cache_site_filters = dict()
        self._cache_snp_sites = None
        self._cache_snp_genotypes = dict()
        self._cache_genome = None
        self._cache_annotator = None
        self._cache_genome_features = dict()
        self._cache_cross_metadata = None
        self._cache_site_annotations = None
        self._cache_cnv_hmm = dict()
        self._cache_cnv_coverage_calls = dict()
        self._cache_cnv_discordant_read_calls = dict()
        self._cache_haplotypes = dict()
        self._cache_haplotype_sites = dict()
        self._cache_cohort_metadata = dict()
        self._cache_sample_metadata = dict()
        self._cache_aim_variants = dict()
        self._cache_locate_site_class = dict()

        if results_cache is not None:
            results_cache = Path(results_cache).expanduser().resolve()
        self._results_cache = results_cache

        # get bokeh to output plots to the notebook - this is a common gotcha,
        # users forget to do this and wonder why bokeh plots don't show
        if bokeh_output_notebook:
            import bokeh.io as bkio

            bkio.output_notebook(hide_banner=True)

        # Occasionally, colab will allocate a VM outside the US, e.g., in
        # Europe or Asia. Because the MalariaGEN data GCS bucket is located
        # in the US, this is usually bad for performance, because of
        # increased latency and lower bandwidth. Add a check for this and
        # issue a warning if not in the US.
        client_details = None
        if check_location:
            try:
                client_details = ipinfo.getHandler().getDetails()
                if GCS_URL in url and colab and client_details.country != "US":
                    warnings.warn(
                        dedent(
                            """
                        Your currently allocated Google Colab VM is not located in the US.
                        This usually means that data access will be substantially slower.
                        If possible, select "Runtime > Disconnect and delete runtime" from
                        the menu to request a new VM and try again.
                    """
                        )
                    )

            except OSError:
                pass
        self._client_details = client_details

    @property
    def _client_location(self):
        details = self._client_details
        if details is not None:
            region = details.region
            country = details.country
            location = f"{region}, {country}"
            if colab:
                location += " (colab)"
            elif hasattr(details, "hostname"):
                hostname = details.hostname
                if hostname.endswith("googleusercontent.com"):
                    location += " (Google Cloud)"
        else:
            location = "unknown"
        return location

    def __repr__(self):
        text = (
            f"<MalariaGEN Ag3 API client>\n"
            f"Storage URL             : {self._url}\n"
            f"Data releases available : {', '.join(self.releases)}\n"
            f"Results cache           : {self._results_cache}\n"
            f"Cohorts analysis        : {self._cohorts_analysis}\n"
            f"Species analysis        : {self._species_analysis}\n"
            f"Site filters analysis   : {self._site_filters_analysis}\n"
            f"Software version        : malariagen_data {malariagen_data.__version__}\n"
            f"Client location         : {self._client_location}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net. For API documentation see \n"
            f"https://malariagen.github.io/vector-data/ag3/api.html"
        )
        return text

    def _repr_html_(self):
        html = f"""
            <table class="malariagen-ag3">
                <thead>
                    <tr>
                        <th style="text-align: left" colspan="2">MalariaGEN Ag3 API client</th>
                    </tr>
                    <tr><td colspan="2" style="text-align: left">
                        Please note that data are subject to terms of use,
                        for more information see <a href="https://www.malariagen.net/data">
                        the MalariaGEN website</a> or contact data@malariagen.net.
                        See also the <a href="https://malariagen.github.io/vector-data/ag3/api.html">Ag3 API docs</a>.
                    </td></tr>
                </thead>
                <tbody>
                    <tr>
                        <th style="text-align: left">
                            Storage URL
                        </th>
                        <td>{self._url}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Data releases available
                        </th>
                        <td>{', '.join(self.releases)}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Results cache
                        </th>
                        <td>{self._results_cache}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Cohorts analysis
                        </th>
                        <td>{self._cohorts_analysis}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Species analysis
                        </th>
                        <td>{self._species_analysis}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Site filters analysis
                        </th>
                        <td>{self._site_filters_analysis}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Software version
                        </th>
                        <td>malariagen_data {malariagen_data.__version__}</td>
                    </tr>
                    <tr>
                        <th style="text-align: left">
                            Client location
                        </th>
                        <td>{self._client_location}</td>
                    </tr>
                </tbody>
            </table>
        """
        return html

    def _progress(self, iterable, **kwargs):
        # progress doesn't mix well with debug logging
        disable = self._debug or not self._show_progress
        return tqdm(iterable, disable=disable, **kwargs)

    def _dask_progress(self, **kwargs):
        disable = not self._show_progress
        return TqdmCallback(disable=disable, **kwargs)

    @property
    def releases(self):
        """The releases for which data are available at the given storage
        location."""
        if self._cache_releases is None:
            if self._pre:
                # Here we discover which releases are available, by listing the storage
                # directory and examining the subdirectories. This may include "pre-releases"
                # where data may be incomplete.
                sub_dirs = [p.split("/")[-1] for p in self._fs.ls(self._base_path)]
                releases = tuple(
                    sorted(
                        [
                            _path_to_release(d)
                            for d in sub_dirs
                            if d.startswith("v3")
                            and self._fs.exists(f"{self._base_path}/{d}/manifest.tsv")
                        ]
                    )
                )
                if len(releases) == 0:
                    raise ValueError("No releases found.")
                self._cache_releases = releases
            else:
                self._cache_releases = self._public_releases
        return self._cache_releases

    def _read_sample_sets(self, *, release):
        """Read the manifest of sample sets for a given release."""
        release_path = _release_to_path(release)
        path = f"{self._base_path}/{release_path}/manifest.tsv"
        with self._fs.open(path) as f:
            df = pd.read_csv(f, sep="\t", na_values="")
        df["release"] = release
        return df

    def sample_sets(self, release=None):
        """Access a dataframe of sample sets.

        Parameters
        ----------
        release : str, optional
            Release identifier. Give "3.0" to access the Ag1000G phase 3 data
            release.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample sets, one row per sample set.

        """

        if release is None:
            # retrieve sample sets from all available releases
            release = self.releases

        if isinstance(release, str):
            # retrieve sample sets for a single release

            if release not in self.releases:
                raise ValueError(f"Release not available: {release!r}")

            try:
                df = self._cache_sample_sets[release]

            except KeyError:
                df = self._read_sample_sets(release=release)
                self._cache_sample_sets[release] = df

        elif isinstance(release, (list, tuple)):

            # check no duplicates
            counter = Counter(release)
            for k, v in counter.items():
                if v > 1:
                    raise ValueError(f"Duplicate values: {k!r}.")

            # retrieve sample sets from multiple releases
            df = pd.concat(
                [self.sample_sets(release=r) for r in release],
                axis=0,
                ignore_index=True,
            )

        else:
            raise TypeError

        return df.copy()

    @property
    def v3_wild(self):
        """Legacy, convenience property to access sample sets from the
        3.0 release, excluding the lab crosses."""
        return [
            x
            for x in self.sample_sets(release="3.0")["sample_set"].tolist()
            if x != "AG1000G-X"
        ]

    def _lookup_release(self, *, sample_set):
        """Find which release a sample set was included in."""

        if self._cache_sample_set_to_release is None:
            df_sample_sets = self.sample_sets().set_index("sample_set")
            self._cache_sample_set_to_release = df_sample_sets["release"].to_dict()

        try:
            return self._cache_sample_set_to_release[sample_set]
        except KeyError:
            raise ValueError(f"No release found for sample set {sample_set!r}")

    def _read_general_metadata(self, *, sample_set):
        """Read metadata for a single sample set."""
        try:
            df = self._cache_general_metadata[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/metadata/general/{sample_set}/samples.meta.csv"
            dtype = {
                "sample_id": object,
                "partner_sample_id": object,
                "contributor": object,
                "country": object,
                "location": object,
                "year": "int64",
                "month": "int64",
                "latitude": "float64",
                "longitude": "float64",
                "sex_call": object,
            }
            with self._fs.open(path) as f:
                df = pd.read_csv(f, na_values="", dtype=dtype)

            # ensure all column names are lower case
            df.columns = [c.lower() for c in df.columns]

            # add a couple of columns for convenience
            df["sample_set"] = sample_set
            df["release"] = release

            self._cache_general_metadata[sample_set] = df
        return df.copy()

    def _read_species_calls(self, *, sample_set):
        """Read species calls for a single sample set."""
        key = sample_set
        try:
            df = self._cache_species_calls[key]

        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path_prefix = f"{self._base_path}/{release_path}/metadata"
            if self._species_analysis == "aim_20220528":
                path = f"{path_prefix}/species_calls_aim_20220528/{sample_set}/samples.species_aim.csv"
                dtype = {
                    "aim_species_gambcolu_arabiensis": object,
                    "aim_species_gambiae_coluzzii": object,
                    "aim_species": object,
                }
                # Specify species_cols in case the file is missing
                species_cols = (
                    "aim_species_fraction_arab",
                    "aim_species_fraction_colu",
                    "aim_species_fraction_colu_no2L",
                    "aim_species_gambcolu_arabiensis",
                    "aim_species_gambiae_coluzzii",
                    "aim_species",
                )
            elif self._species_analysis == "aim_20200422":
                # TODO this is legacy, deprecate at some point
                path = f"{path_prefix}/species_calls_20200422/{sample_set}/samples.species_aim.csv"
                dtype = {
                    "species_gambcolu_arabiensis": object,
                    "species_gambiae_coluzzii": object,
                }
                # Specify species_cols in case the file is missing
                # N.B., these legacy column prefixes will be normalised downstream
                species_cols = (
                    "aim_fraction_colu",
                    "aim_fraction_arab",
                    "species_gambcolu_arabiensis",
                    "species_gambiae_coluzzii",
                )
            elif self._species_analysis == "pca_20200422":
                # TODO this is legacy, deprecate at some point
                path = f"{path_prefix}/species_calls_20200422/{sample_set}/samples.species_pca.csv"
                dtype = {
                    "species_gambcolu_arabiensis": object,
                    "species_gambiae_coluzzii": object,
                }
                # Specify species_cols in case the file is missing
                # N.B., these legacy column prefixes will be normalised downstream
                species_cols = (
                    "PC1",
                    "PC2",
                    "species_gambcolu_arabiensis",
                    "species_gambiae_coluzzii",
                )
            else:
                raise ValueError(
                    f"Unknown species calling analysis: {self._species_analysis!r}"
                )

            # N.B., species calls do not always exist, need to handle FileNotFoundError
            try:
                with self._fs.open(path) as f:
                    df = pd.read_csv(
                        f,
                        na_values=["", "NA"],
                        # ensure correct dtype even where all values are missing
                        dtype=dtype,
                    )
            except FileNotFoundError:
                # Get sample ids as an index via general metadata (has caching)
                df_general = self._read_general_metadata(sample_set=sample_set)
                df_general.set_index("sample_id", inplace=True)

                # Create a blank DataFrame with species_cols and sample_id index
                df = pd.DataFrame(columns=species_cols, index=df_general.index.copy())

                # Revert sample_id index to column
                df.reset_index(inplace=True)

            # add a single species call column, for convenience
            def consolidate_species(s):
                species_gambcolu_arabiensis = s["species_gambcolu_arabiensis"]
                species_gambiae_coluzzii = s["species_gambiae_coluzzii"]
                if species_gambcolu_arabiensis == "arabiensis":
                    return "arabiensis"
                elif species_gambcolu_arabiensis == "intermediate":
                    return "intermediate_arabiensis_gambiae"
                elif species_gambcolu_arabiensis == "gamb_colu":
                    # look at gambiae_vs_coluzzii
                    if species_gambiae_coluzzii == "gambiae":
                        return "gambiae"
                    elif species_gambiae_coluzzii == "coluzzii":
                        return "coluzzii"
                    elif species_gambiae_coluzzii == "intermediate":
                        return "intermediate_gambiae_coluzzii"
                else:
                    # some individuals, e.g., crosses, have a missing species call
                    return np.nan

            if self._species_analysis == "aim_20200422":
                # TODO this is legacy, deprecate at some point
                df["species"] = df.apply(consolidate_species, axis=1)
                # normalise column prefixes
                df = df.rename(
                    columns={
                        "aim_fraction_arab": "aim_species_fraction_arab",
                        "aim_fraction_colu": "aim_species_fraction_colu",
                        "species_gambcolu_arabiensis": "aim_species_gambcolu_arabiensis",
                        "species_gambiae_coluzzii": "aim_species_gambiae_coluzzii",
                        "species": "aim_species",
                    }
                )
            elif self._species_analysis == "pca_20200422":
                # TODO this is legacy, deprecate at some point
                df["species"] = df.apply(consolidate_species, axis=1)
                # normalise column prefixes
                df = df.rename(
                    # normalise column prefixes
                    columns={
                        "PC1": "pca_species_PC1",
                        "PC2": "pca_species_PC2",
                        "species_gambcolu_arabiensis": "pca_species_gambcolu_arabiensis",
                        "species_gambiae_coluzzii": "pca_species_gambiae_coluzzii",
                        "species": "pca_species",
                    }
                )

            # ensure all column names are lower case
            df.columns = [c.lower() for c in df.columns]

            self._cache_species_calls[key] = df

        return df.copy()

    def _prep_sample_sets_arg(self, *, sample_sets):
        """Common handling for the `sample_sets` parameter. For convenience, we
        allow this to be a single sample set, or a list of sample sets, or a
        release identifier, or a list of release identifiers."""

        if sample_sets is None:
            # all available sample sets
            sample_sets = self.sample_sets()["sample_set"].tolist()

        elif isinstance(sample_sets, str):

            if sample_sets.startswith("3."):
                # convenience, can use a release identifier to denote all sample sets in a release
                sample_sets = self.sample_sets(release=sample_sets)[
                    "sample_set"
                ].tolist()

            else:
                # single sample set, normalise to always return a list
                sample_sets = [sample_sets]

        elif isinstance(sample_sets, (list, tuple)):
            # list or tuple of sample sets or releases
            prepped_sample_sets = []
            for s in sample_sets:

                # make a recursive call to handle the case where s is a release identifier
                sp = self._prep_sample_sets_arg(sample_sets=s)

                # make sure we end up with a flat list of sample sets
                if isinstance(sp, str):
                    prepped_sample_sets.append(sp)
                else:
                    prepped_sample_sets.extend(sp)
            sample_sets = prepped_sample_sets

        else:
            raise TypeError(
                f"Invalid type for sample_sets parameter; expected str, list or tuple; found: {sample_sets!r}"
            )

        # check all sample sets selected at most once
        counter = Counter(sample_sets)
        for k, v in counter.items():
            if v > 1:
                raise ValueError(
                    f"Bad value for sample_sets parameter, {k:!r} selected more than once."
                )

        return sample_sets

    def species_calls(self, sample_sets=None):
        """Access species calls for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"] or a
            release identifier (e.g., "3.0") or a list of release identifiers.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of species calls for one or more sample sets, one row
            per sample.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [self._read_species_calls(sample_set=s) for s in sample_sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def _sample_metadata(self, *, sample_set):
        df = self._read_general_metadata(sample_set=sample_set)
        df_species = self._read_species_calls(sample_set=sample_set)
        df = df.merge(df_species, on="sample_id", sort=False)
        df_cohorts = self._read_cohort_metadata(sample_set=sample_set)
        df = df.merge(df_cohorts, on="sample_id", sort=False)
        return df

    def sample_metadata(
        self,
        sample_sets=None,
        sample_query=None,
    ):
        """Access sample metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".

        Returns
        -------
        df_samples : pandas.DataFrame
            A dataframe of sample metadata, one row per sample.

        """

        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        cache_key = tuple(sample_sets)

        try:

            df_samples = self._cache_sample_metadata[cache_key]

        except KeyError:

            # concatenate multiple sample sets
            dfs = []
            # there can be some delay here due to network latency, so show progress
            sample_sets_iterator = self._progress(
                sample_sets, desc="Load sample metadata"
            )
            for s in sample_sets_iterator:
                df = self._sample_metadata(sample_set=s)
                dfs.append(df)
            df_samples = pd.concat(dfs, axis=0, ignore_index=True)
            self._cache_sample_metadata[cache_key] = df_samples

        # for convenience, apply a query
        if sample_query is not None:
            df_samples = df_samples.query(sample_query).reset_index(drop=True)

        return df_samples.copy()

    def open_site_filters(self, mask):
        """Open site filters zarr.

        Parameters
        ----------
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_site_filters[mask]
        except KeyError:
            path = f"{self._base_path}/v3/site_filters/{self._site_filters_analysis}/{mask}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[mask] = root
            return root

    def _site_filters(
        self,
        *,
        region,
        mask,
        field,
        inline_array,
        chunks,
    ):
        assert isinstance(region, Region)
        root = self.open_site_filters(mask=mask)
        z = root[f"{region.contig}/variants/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            root = self.open_snp_sites()
            pos = root[f"{region.contig}/variants/POS"][:]
            loc_region = locate_region(region, pos)
            d = d[loc_region]
        return d

    def site_filters(
        self,
        region,
        mask,
        field="filter_pass",
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site filters.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Mask to use.
        field : str, optional
            Array to access.
        inline_array : bool, optional
            Passed through to dask.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of boolean values identifying sites that pass the filters.

        """

        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        d = da.concatenate(
            [
                self._site_filters(
                    region=r,
                    mask=mask,
                    field=field,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                for r in region
            ]
        )

        return d

    def open_snp_sites(self):
        """Open SNP sites zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        if self._cache_snp_sites is None:
            path = f"{self._base_path}/v3/snp_genotypes/all/sites/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    def _snp_sites(
        self,
        *,
        region,
        field,
        inline_array,
        chunks,
    ):
        assert isinstance(region, Region), type(region)
        root = self.open_snp_sites()
        z = root[f"{region.contig}/variants/{field}"]
        ret = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            if field == "POS":
                pos = z[:]
            else:
                pos = root[f"{region.contig}/variants/POS"][:]
            loc_region = locate_region(region, pos)
            ret = ret[loc_region]
        return ret

    def snp_sites(
        self,
        region,
        field,
        site_mask=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        field : {"POS", "REF", "ALT"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of either SNP positions, reference alleles or alternate
            alleles.

        """
        debug = self._log.debug

        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access SNP sites and concatenate over regions")
        ret = da.concatenate(
            [
                self._snp_sites(
                    region=r,
                    field=field,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for r in region
            ],
            axis=0,
        )

        debug("apply site mask if requested")
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region,
                mask=site_mask,
                chunks=chunks,
                inline_array=inline_array,
            )
            ret = da_compress(loc_sites, ret, axis=0)

        return ret

    def open_snp_genotypes(self, sample_set):
        """Open SNP genotypes zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_snp_genotypes[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_genotypes/all/{sample_set}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_genotypes[sample_set] = root
            return root

    def _snp_genotypes(self, *, region, sample_set, field, inline_array, chunks):
        """Access SNP genotypes for a single contig and a single sample set."""
        assert isinstance(region, Region)
        assert isinstance(sample_set, str)
        root = self.open_snp_genotypes(sample_set=sample_set)
        z = root[f"{region.contig}/calldata/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            pos = self.snp_sites(region=region.contig, field="POS")
            loc_region = locate_region(region, pos)
            d = d[loc_region]

        return d

    def snp_genotypes(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        field="GT",
        site_mask=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP genotypes and associated data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of either genotypes (GT), genotype quality (GQ), allele
            depths (AD) or mapping quality (MQ) values.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = self.resolve_region(region)

        debug("normalise region to list to simplify concatenation logic")
        if isinstance(region, Region):
            region = [region]

        debug("concatenate multiple sample sets and/or contigs")
        lx = []
        for r in region:
            ly = []

            for s in sample_sets:
                y = self._snp_genotypes(
                    region=Region(r.contig, None, None),
                    sample_set=s,
                    field=field,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            debug("concatenate data from multiple sample sets")
            x = da.concatenate(ly, axis=1)

            debug("locate region - do this only once, optimisation")
            if r.start or r.end:
                pos = self.snp_sites(region=r.contig, field="POS")
                loc_region = locate_region(r, pos)
                x = x[loc_region]

            lx.append(x)

        debug("concatenate data from multiple regions")
        d = da.concatenate(lx, axis=0)

        debug("apply site filters if requested")
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=region,
                mask=site_mask,
            )
            d = da_compress(loc_sites, d, axis=0)

        debug("apply sample query if requested")
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            d = da.compress(loc_samples, d, axis=1)

        return d

    def open_genome(self):
        """Open the reference genome zarr.

        Returns
        -------
        root : zarr.hierarchy.Group
            Zarr hierarchy containing the reference genome sequence.

        """
        if self._cache_genome is None:
            path = f"{self._base_path}/{GENOME_ZARR_PATH}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_genome = zarr.open_consolidated(store=store)
        return self._cache_genome

    def genome_sequence(self, region, inline_array=True, chunks="native"):
        """Access the reference genome sequence.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array
            An array of nucleotides giving the reference genome sequence for the
            given contig.

        """
        genome = self.open_genome()
        region = self.resolve_region(region)
        z = genome[region.contig]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)

        if region.start:
            slice_start = region.start - 1
        else:
            slice_start = None
        if region.end:
            slice_stop = region.end
        else:
            slice_stop = None
        loc_region = slice(slice_start, slice_stop)

        return d[loc_region]

    def geneset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to genome_features()."""
        return self.genome_features(*args, **kwargs)

    def genome_features(
        self, region=None, attributes=("ID", "Parent", "Name", "description")
    ):
        """Access genome feature annotations (AgamP4.12).

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all
            attributes.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of genome annotations, one row per feature.

        """
        debug = self._log.debug

        if attributes is not None:
            attributes = tuple(attributes)

        try:
            df = self._cache_genome_features[attributes]

        except KeyError:
            path = f"{self._base_path}/{self._genome_feature_GFF_path}"
            with self._fs.open(path, mode="rb") as f:
                df = read_gff3(f, compression="gzip")
            if attributes is not None:
                df = unpack_gff3_attributes(df, attributes=attributes)
            self._cache_genome_features[attributes] = df

        debug("handle region")
        if region is not None:

            region = self.resolve_region(region)

            debug("normalise to list to simplify concatenation logic")
            if isinstance(region, Region):
                region = [region]

            debug("apply region query")
            parts = []
            for r in region:
                df_part = df.query(f"contig == '{r.contig}'")
                if r.end is not None:
                    df_part = df_part.query(f"start <= {r.end}")
                if r.start is not None:
                    df_part = df_part.query(f"end >= {r.start}")
                parts.append(df_part)
            df = pd.concat(parts, axis=0)

        return df.reset_index(drop=True).copy()

    def _transcript_to_gene_name(self, transcript):
        df_genome_features = self.genome_features().set_index("ID")
        rec_transcript = df_genome_features.loc[transcript]
        parent = rec_transcript["Parent"]
        rec_parent = df_genome_features.loc[parent]

        # manual overrides
        if parent == "AGAP004707":
            parent_name = "Vgsc/para"
        else:
            parent_name = rec_parent["Name"]

        return parent_name

    def is_accessible(
        self,
        region,
        site_mask,
    ):
        """Compute genome accessibility array.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.

        Returns
        -------
        a : numpy.ndarray
            An array of boolean values identifying accessible genome sites.

        """
        debug = self._log.debug

        debug("resolve region")
        region = self.resolve_region(region)

        debug("determine contig sequence length")
        seq_length = self.genome_sequence(region).shape[0]

        debug("set up output")
        is_accessible = np.zeros(seq_length, dtype=bool)

        pos = self.snp_sites(region=region, field="POS").compute()
        if region.start:
            offset = region.start
        else:
            offset = 1

        debug("access site filters")
        filter_pass = self.site_filters(
            region=region,
            mask=site_mask,
        ).compute()

        debug("assign values from site filters")
        is_accessible[pos - offset] = filter_pass

        return is_accessible

    def _site_mask_ids(self):
        if self._site_filters_analysis == "dt_20200416":
            return "gamb_colu_arab", "gamb_colu", "arab"
        else:
            raise ValueError

    def _snp_df(self, *, transcript):
        """Set up a dataframe with SNP site and filter columns."""
        debug = self._log.debug

        debug("get feature direct from genome_features")
        gs = self.genome_features()
        feature = gs[gs["ID"] == transcript].squeeze()
        contig = feature.contig
        region = Region(contig, feature.start, feature.end)

        debug("grab pos, ref and alt for chrom arm from snp_sites")
        pos = self.snp_sites(region=contig, field="POS")
        ref = self.snp_sites(region=contig, field="REF")
        alt = self.snp_sites(region=contig, field="ALT")
        loc_feature = locate_region(region, pos)
        pos = pos[loc_feature].compute()
        ref = ref[loc_feature].compute()
        alt = alt[loc_feature].compute()

        debug("access site filters")
        filter_pass = dict()
        masks = self._site_mask_ids()
        for m in masks:
            x = self.site_filters(region=contig, mask=m)
            x = x[loc_feature].compute()
            filter_pass[m] = x

        debug("set up columns with contig, pos, ref, alt columns")
        cols = {
            "contig": contig,
            "position": np.repeat(pos, 3),
            "ref_allele": np.repeat(ref.astype("U1"), 3),
            "alt_allele": alt.astype("U1").flatten(),
        }

        debug("add mask columns")
        for m in masks:
            x = filter_pass[m]
            cols[f"pass_{m}"] = np.repeat(x, 3)

        debug("construct dataframe")
        df_snps = pd.DataFrame(cols)

        return region, df_snps

    def _annotator(self):
        """Set up variant effect annotator."""
        if self._cache_annotator is None:
            self._cache_annotator = veff.Annotator(
                genome=self.open_genome(), genome_features=self.genome_features()
            )
        return self._cache_annotator

    def snp_effects(
        self,
        transcript,
        site_mask=None,
    ):
        """Compute variant effects for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}, optional
            Site filters mask to apply.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of all possible SNP variants and their effects, one row
            per variant.

        """
        debug = self._log.debug

        debug("setup initial dataframe of SNPs")
        _, df_snps = self._snp_df(transcript=transcript)

        debug("setup variant effect annotator")
        ann = self._annotator()

        debug("apply mask if requested")
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        debug("reset index after filtering")
        df_snps.reset_index(inplace=True, drop=True)

        debug("add effects to the dataframe")
        ann.get_effects(transcript=transcript, variants=df_snps)

        return df_snps

    def snp_allele_frequencies(
        self,
        transcript,
        cohorts,
        sample_query=None,
        min_cohort_size=10,
        site_mask=None,
        sample_sets=None,
        drop_invariant=True,
        effects=True,
    ):
        """Compute per variant allele frequencies for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RD".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int
            Minimum cohort size. Any cohorts below this size are omitted.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are
            dropped from the result.
        effects : bool, optional
            If True, add SNP effect columns.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of SNP frequencies, one row per variant.

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from
        output.

        """
        debug = self._log.debug

        debug("check parameters")
        _check_param_min_cohort_size(min_cohort_size)

        debug("access sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("setup initial dataframe of SNPs")
        region, df_snps = self._snp_df(transcript=transcript)

        debug("get genotypes")
        gt = self.snp_genotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            field="GT",
        )

        debug("slice to feature location")
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        debug("build coh dict")
        coh_dict = _locate_cohorts(cohorts=cohorts, df_samples=df_samples)

        debug("count alleles")
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            n_samples = np.count_nonzero(loc_coh)
            debug(f"{coh}, {n_samples} samples")
            if n_samples >= min_cohort_size:
                gt_coh = np.compress(loc_coh, gt, axis=1)
                ac_coh = allel.GenotypeArray(gt_coh).count_alleles(max_allele=3)
                af_coh = ac_coh.to_frequencies()
                freq_cols["frq_" + coh] = af_coh[:, 1:].flatten()

        debug("build a dataframe with the frequency columns")
        df_freqs = pd.DataFrame(freq_cols)

        debug("compute max_af")
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        debug("build the final dataframe")
        df_snps.reset_index(drop=True, inplace=True)
        df_snps = pd.concat([df_snps, df_freqs, df_max_af], axis=1)

        debug("apply site mask if requested")
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        debug("drop invariants")
        if drop_invariant:
            loc_variant = df_snps["max_af"] > 0
            df_snps = df_snps.loc[loc_variant]

        debug("reset index after filtering")
        df_snps.reset_index(inplace=True, drop=True)

        if effects:

            debug("add effect annotations")
            ann = self._annotator()
            ann.get_effects(
                transcript=transcript, variants=df_snps, progress=self._progress
            )

            debug("add label")
            df_snps["label"] = _pandas_apply(
                _make_snp_label_effect,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
            )

            debug("set index")
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele", "aa_change"],
                inplace=True,
            )

        else:

            debug("add label")
            df_snps["label"] = _pandas_apply(
                _make_snp_label,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele"],
            )

            debug("set index")
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele"],
                inplace=True,
            )

        debug("add dataframe metadata")
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_snps.attrs["title"] = title

        return df_snps

    def cross_metadata(self):
        """Load a dataframe containing metadata about samples in colony crosses,
        including which samples are parents or progeny in which crosses.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata for colony crosses.

        """
        debug = self._log.debug

        if self._cache_cross_metadata is None:

            path = f"{self._base_path}/v3/metadata/crosses/crosses.fam"
            fam_names = [
                "cross",
                "sample_id",
                "father_id",
                "mother_id",
                "sex",
                "phenotype",
            ]
            with self._fs.open(path) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    na_values=["", "0"],
                    names=fam_names,
                    dtype={"sex": str},
                )

            debug("convert 'sex' column for consistency with sample metadata")
            df.loc[df["sex"] == "1", "sex"] = "M"
            df.loc[df["sex"] == "2", "sex"] = "F"

            debug("add a 'role' column for convenience")
            df["role"] = "progeny"
            df.loc[df["mother_id"].isna(), "role"] = "parent"

            debug("drop 'phenotype' column, not used")
            df.drop("phenotype", axis="columns", inplace=True)

            self._cache_cross_metadata = df

        return self._cache_cross_metadata.copy()

    def open_site_annotations(self):
        """Open site annotations zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """

        if self._cache_site_annotations is None:
            path = f"{self._base_path}/reference/genome/agamp4/Anopheles-gambiae-PEST_SEQANNOTATION_AgamP4.12.zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_site_annotations = zarr.open_consolidated(store=store)
        return self._cache_site_annotations

    def site_annotations(
        self,
        region,
        site_mask=None,
        inline_array=True,
        chunks="auto",
    ):
        """Load site annotations.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        inline_array : bool, optional
            Passed through to dask.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of site annotations.

        """
        # N.B., we default to chunks="auto" here for performance reasons

        debug = self._log.debug

        debug("resolve region")
        region = self.resolve_region(region)
        if isinstance(region, list):
            raise TypeError("Multiple regions not supported.")
        contig = region.contig

        debug("open site annotations zarr")
        root = self.open_site_annotations()

        debug("build a dataset")
        ds = xr.Dataset()
        for field in (
            "codon_degeneracy",
            "codon_nonsyn",
            "codon_position",
            "seq_cls",
            "seq_flen",
            "seq_relpos_start",
            "seq_relpos_stop",
        ):
            data = da_from_zarr(
                root[field][contig],
                inline_array=inline_array,
                chunks=chunks,
            )
            ds[field] = "variants", data

        debug("subset to SNP positions")
        pos = self.snp_sites(
            region=contig,
            field="POS",
            site_mask=site_mask,
            inline_array=inline_array,
            chunks=chunks,
        )
        pos = pos.compute()
        if region.start or region.end:
            loc_region = locate_region(region, pos)
            pos = pos[loc_region]
        idx = pos - 1
        ds = ds.isel(variants=idx)

        return ds

    def _locate_site_class(
        self,
        *,
        region,
        site_mask,
        site_class,
    ):
        debug = self._log.debug

        # cache these data in memory to avoid repeated computation
        cache_key = (region, site_mask, site_class)

        try:
            loc_ann = self._cache_locate_site_class[cache_key]

        except KeyError:
            debug("access site annotations data")
            ds_ann = self.site_annotations(
                region=region,
                site_mask=site_mask,
            )
            codon_pos = ds_ann["codon_position"].data
            codon_deg = ds_ann["codon_degeneracy"].data
            seq_cls = ds_ann["seq_cls"].data
            seq_flen = ds_ann["seq_flen"].data
            seq_relpos_start = ds_ann["seq_relpos_start"].data
            seq_relpos_stop = ds_ann["seq_relpos_stop"].data
            site_class = site_class.upper()

            debug("define constants used in site annotations data")
            SEQ_CLS_UNKNOWN = 0  # noqa
            SEQ_CLS_UPSTREAM = 1
            SEQ_CLS_DOWNSTREAM = 2
            SEQ_CLS_5UTR = 3
            SEQ_CLS_3UTR = 4
            SEQ_CLS_CDS_FIRST = 5
            SEQ_CLS_CDS_MID = 6
            SEQ_CLS_CDS_LAST = 7
            SEQ_CLS_INTRON_FIRST = 8
            SEQ_CLS_INTRON_MID = 9
            SEQ_CLS_INTRON_LAST = 10
            CODON_DEG_UNKNOWN = 0  # noqa
            CODON_DEG_0 = 1
            CODON_DEG_2_SIMPLE = 2
            CODON_DEG_2_COMPLEX = 3  # noqa
            CODON_DEG_4 = 4

            debug("set up site selection")

            if site_class == "CDS_DEG_4":
                # 4-fold degenerate coding sites
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_CDS_FIRST)
                        | (seq_cls == SEQ_CLS_CDS_MID)
                        | (seq_cls == SEQ_CLS_CDS_LAST)
                    )
                    & (codon_pos == 2)
                    & (codon_deg == CODON_DEG_4)
                )

            elif site_class == "CDS_DEG_2_SIMPLE":
                # 2-fold degenerate coding sites
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_CDS_FIRST)
                        | (seq_cls == SEQ_CLS_CDS_MID)
                        | (seq_cls == SEQ_CLS_CDS_LAST)
                    )
                    & (codon_pos == 2)
                    & (codon_deg == CODON_DEG_2_SIMPLE)
                )

            elif site_class == "CDS_DEG_0":
                # non-degenerate coding sites
                loc_ann = (
                    (seq_cls == SEQ_CLS_CDS_FIRST)
                    | (seq_cls == SEQ_CLS_CDS_MID)
                    | (seq_cls == SEQ_CLS_CDS_LAST)
                ) & (codon_deg == CODON_DEG_0)

            elif site_class == "INTRON_SHORT":
                # short introns, excluding splice regions
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_INTRON_FIRST)
                        | (seq_cls == SEQ_CLS_INTRON_MID)
                        | (seq_cls == SEQ_CLS_INTRON_LAST)
                    )
                    & (seq_flen < 100)
                    & (seq_relpos_start > 10)
                    & (seq_relpos_stop > 10)
                )

            elif site_class == "INTRON_LONG":
                # long introns, excluding splice regions
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_INTRON_FIRST)
                        | (seq_cls == SEQ_CLS_INTRON_MID)
                        | (seq_cls == SEQ_CLS_INTRON_LAST)
                    )
                    & (seq_flen > 200)
                    & (seq_relpos_start > 10)
                    & (seq_relpos_stop > 10)
                )

            elif site_class == "INTRON_SPLICE_5PRIME":
                # 5' intron splice regions
                loc_ann = (
                    (seq_cls == SEQ_CLS_INTRON_FIRST)
                    | (seq_cls == SEQ_CLS_INTRON_MID)
                    | (seq_cls == SEQ_CLS_INTRON_LAST)
                ) & (seq_relpos_start < 2)

            elif site_class == "INTRON_SPLICE_3PRIME":
                # 3' intron splice regions
                loc_ann = (
                    (seq_cls == SEQ_CLS_INTRON_FIRST)
                    | (seq_cls == SEQ_CLS_INTRON_MID)
                    | (seq_cls == SEQ_CLS_INTRON_LAST)
                ) & (seq_relpos_stop < 2)

            elif site_class == "UTR_5PRIME":
                # 5' UTR
                loc_ann = seq_cls == SEQ_CLS_5UTR

            elif site_class == "UTR_3PRIME":
                # 3' UTR
                loc_ann = seq_cls == SEQ_CLS_3UTR

            elif site_class == "INTERGENIC":
                # intergenic regions, distant from a gene
                loc_ann = (
                    (seq_cls == SEQ_CLS_UPSTREAM) & (seq_relpos_stop > 10_000)
                ) | ((seq_cls == SEQ_CLS_DOWNSTREAM) & (seq_relpos_start > 10_000))

            else:
                raise NotImplementedError(site_class)

            debug("compute site selection")
            with self._dask_progress(desc=f"Locate {site_class} sites"):
                loc_ann = loc_ann.compute()

            self._cache_locate_site_class[cache_key] = loc_ann

        return loc_ann

    def _snp_variants_dataset(self, *, contig, inline_array, chunks):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("variant arrays")
        sites_root = self.open_snp_sites()

        debug("variant_position")
        pos_z = sites_root[f"{contig}/variants/POS"]
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        debug("variant_allele")
        ref_z = sites_root[f"{contig}/variants/REF"]
        alt_z = sites_root[f"{contig}/variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        debug("variant_contig")
        contig_index = self.contigs.index(contig)
        variant_contig = da.full_like(
            variant_position, fill_value=contig_index, dtype="u1"
        )
        coords["variant_contig"] = [DIM_VARIANT], variant_contig

        debug("site filters arrays")
        for mask in self._site_mask_ids():
            filters_root = self.open_site_filters(mask=mask)
            z = filters_root[f"{contig}/variants/filter_pass"]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"variant_filter_pass_{mask}"] = [DIM_VARIANT], d

        debug("set up attributes")
        attrs = {"contigs": self.contigs}

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def _snp_calls_dataset(self, *, contig, sample_set, inline_array, chunks):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("call arrays")
        calls_root = self.open_snp_genotypes(sample_set=sample_set)
        gt_z = calls_root[f"{contig}/calldata/GT"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        gq_z = calls_root[f"{contig}/calldata/GQ"]
        call_gq = da_from_zarr(gq_z, inline_array=inline_array, chunks=chunks)
        ad_z = calls_root[f"{contig}/calldata/AD"]
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        mq_z = calls_root[f"{contig}/calldata/MQ"]
        call_mq = da_from_zarr(mq_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_GQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_gq)
        data_vars["call_MQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_mq)
        data_vars["call_AD"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE],
            call_ad,
        )

        debug("sample arrays")
        z = calls_root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        # decode to str, as it is stored as bytes objects
        sample_id = sample_id.astype("U")
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        return ds

    def snp_variants(
        self,
        region,
        site_mask=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP sites and site filters.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing SNP sites and site filters.

        """
        debug = self._log.debug

        debug("normalise parameters")
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access SNP data and concatenate multiple regions")
        lx = []
        for r in region:

            debug("access variants")
            x = self._snp_variants_dataset(
                contig=r.contig,
                inline_array=inline_array,
                chunks=chunks,
            )

            debug("handle region")
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        debug("apply site filters")
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        return ds

    def snp_calls(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask=None,
        site_class=None,
        inline_array=True,
        chunks="native",
        cohort_size=None,
        random_seed=42,
    ):
        """Access SNP sites, site filters and genotype calls.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing SNP sites, site filters and genotype calls.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access SNP calls and concatenate multiple sample sets and/or regions")
        lx = []
        for r in region:

            ly = []
            for s in sample_sets:
                y = self._snp_calls_dataset(
                    contig=r.contig,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            debug("concatenate data from multiple sample sets")
            x = xarray_concat(ly, dim=DIM_SAMPLE)

            debug("add variants variables")
            v = self._snp_variants_dataset(
                contig=r.contig, inline_array=inline_array, chunks=chunks
            )
            x = xr.merge([v, x], compat="override", join="override")

            debug("handle site class")
            if site_class is not None:
                loc_ann = self._locate_site_class(
                    region=r.contig,
                    site_class=site_class,
                    site_mask=None,
                )
                x = x.isel(variants=loc_ann)

            debug("handle region, do this only once - optimisation")
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        debug("apply site filters")
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        debug("add call_genotype_mask")
        ds["call_genotype_mask"] = ds["call_genotype"] < 0

        debug("handle sample query")
        if sample_query is not None:
            if isinstance(sample_query, str):
                df_samples = self.sample_metadata(sample_sets=sample_sets)
                loc_samples = df_samples.eval(sample_query).values
                if np.count_nonzero(loc_samples) == 0:
                    raise ValueError(f"No samples found for query {sample_query!r}")
            else:
                # assume sample query is an indexer, e.g., a list of integers
                loc_samples = sample_query
            ds = ds.isel(samples=loc_samples)

        debug("handle cohort size")
        if cohort_size is not None:
            n_samples = ds.dims["samples"]
            if n_samples < cohort_size:
                raise ValueError(
                    f"not enough samples ({n_samples}) for cohort size ({cohort_size})"
                )
            rng = np.random.default_rng(seed=random_seed)
            loc_downsample = rng.choice(n_samples, size=cohort_size, replace=False)
            loc_downsample.sort()
            ds = ds.isel(samples=loc_downsample)

        return ds

    def snp_dataset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to snp_calls()."""
        return self.snp_calls(*args, **kwargs)

    def open_cnv_hmm(self, sample_set):
        """Open CNV HMM zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_cnv_hmm[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/hmm/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_hmm[sample_set] = root
        return root

    def _cnv_hmm_dataset(self, *, contig, sample_set, inline_array, chunks):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("open zarr")
        root = self.open_cnv_hmm(sample_set=sample_set)

        debug("variant arrays")
        pos = root[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )
        coords["variant_end"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/END"], inline_array=inline_array, chunks=chunks
            ),
        )

        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )

        debug("call arrays")
        data_vars["call_CN"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/CN"], inline_array=inline_array, chunks=chunks
            ),
        )
        data_vars["call_RawCov"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/RawCov"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )
        data_vars["call_NormCov"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/NormCov"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )

        debug("sample arrays")
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )
        for field in "sample_coverage_variance", "sample_is_high_variance":
            data_vars[field] = (
                [DIM_SAMPLE],
                da_from_zarr(root[field], inline_array=inline_array, chunks=chunks),
            )

        debug("set up attributes")
        attrs = {"contigs": self.contigs}

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_hmm(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV HMM data from CNV calling.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of CNV HMM calls and associated data.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access CNV HMM data and concatenate as needed")
        lx = []
        for r in region:

            ly = []
            for s in sample_sets:
                y = self._cnv_hmm_dataset(
                    contig=r.contig,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            debug("concatenate data from multiple sample sets")
            x = xarray_concat(ly, dim=DIM_SAMPLE)

            debug("handle region, do this only once - optimisation")
            if r.start is not None or r.end is not None:
                start = x["variant_position"].values
                end = x["variant_end"].values
                index = pd.IntervalIndex.from_arrays(start, end, closed="both")
                # noinspection PyArgumentList
                other = pd.Interval(r.start, r.end, closed="both")
                loc_region = index.overlaps(other)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        debug("handle sample query")
        if sample_query is not None:

            debug("load sample metadata")
            df_samples = self.sample_metadata(sample_sets=sample_sets)

            debug("align sample metadata with CNV data")
            cnv_samples = ds["sample_id"].values.tolist()
            df_samples_cnv = (
                df_samples.set_index("sample_id").loc[cnv_samples].reset_index()
            )

            debug("apply the query")
            loc_query_samples = df_samples_cnv.eval(sample_query).values
            if np.count_nonzero(loc_query_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")

            ds = ds.isel(samples=loc_query_samples)

        debug("handle coverage variance filter")
        if max_coverage_variance is not None:
            cov_var = ds["sample_coverage_variance"].values
            loc_pass_samples = cov_var <= max_coverage_variance
            ds = ds.isel(samples=loc_pass_samples)

        return ds

    def open_cnv_coverage_calls(self, sample_set, analysis):
        """Open CNV coverage calls zarr.

        Parameters
        ----------
        sample_set : str
        analysis : {'gamb_colu', 'arab', 'crosses'}

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        key = (sample_set, analysis)
        try:
            return self._cache_cnv_coverage_calls[key]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/coverage_calls/{analysis}/zarr"
            # N.B., not all sample_set/analysis combinations exist, need to check
            marker = path + "/.zmetadata"
            if not self._fs.exists(marker):
                raise ValueError(
                    f"analysis f{analysis!r} not implemented for sample set {sample_set!r}"
                )
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_coverage_calls[key] = root
        return root

    def _cnv_coverage_calls_dataset(
        self,
        *,
        contig,
        sample_set,
        analysis,
        inline_array,
        chunks,
    ):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("open zarr")
        root = self.open_cnv_coverage_calls(sample_set=sample_set, analysis=analysis)

        debug("variant arrays")
        pos = root[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )
        coords["variant_end"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/END"], inline_array=inline_array, chunks=chunks
            ),
        )
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )
        coords["variant_id"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/ID"], inline_array=inline_array, chunks=chunks
            ),
        )
        data_vars["variant_CIPOS"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/CIPOS"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )
        data_vars["variant_CIEND"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/CIEND"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )
        data_vars["variant_filter_pass"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/FILTER_PASS"],
                inline_array=inline_array,
                chunks=chunks,
            ),
        )

        debug("call arrays")
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/GT"], inline_array=inline_array, chunks=chunks
            ),
        )

        debug("sample arrays")
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )

        debug("set up attributes")
        attrs = {"contigs": self.contigs}

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_coverage_calls(
        self,
        region,
        sample_set,
        analysis,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV HMM data from genome-wide CNV discovery and filtering.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_set : str
            Sample set identifier.
        analysis : {'gamb_colu', 'arab', 'crosses'}
            Name of CNV analysis.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of CNV alleles and genotypes.

        """
        debug = self._log.debug

        # N.B., we cannot concatenate multiple sample sets here, because
        # different sample sets may have different sets of alleles, as the
        # calling is done independently in different sample sets.

        debug("normalise parameters")
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access data and concatenate as needed")
        lx = []
        for r in region:

            debug("obtain coverage calls for the contig")
            x = self._cnv_coverage_calls_dataset(
                contig=r.contig,
                sample_set=sample_set,
                analysis=analysis,
                inline_array=inline_array,
                chunks=chunks,
            )

            debug("select region")
            if r.start is not None or r.end is not None:
                start = x["variant_position"].values
                end = x["variant_end"].values
                index = pd.IntervalIndex.from_arrays(start, end, closed="both")
                # noinspection PyArgumentList
                other = pd.Interval(r.start, r.end, closed="both")
                loc_region = index.overlaps(other)
                x = x.isel(variants=loc_region)

            lx.append(x)
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def open_cnv_discordant_read_calls(self, sample_set):
        """Open CNV discordant read calls zarr.

        Parameters
        ----------
        sample_set : str

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_cnv_discordant_read_calls[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/discordant_read_calls/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_cnv_discordant_read_calls[sample_set] = root
        return root

    def _cnv_discordant_read_calls_dataset(
        self, *, contig, sample_set, inline_array, chunks
    ):
        debug = self._log.debug

        coords = dict()
        data_vars = dict()

        debug("open zarr")
        root = self.open_cnv_discordant_read_calls(sample_set=sample_set)

        # not all contigs have CNVs, need to check
        # TODO consider returning dataset with zero length variants dimension, would
        # probably simplify downstream logic
        if contig not in root:
            raise ValueError(f"no CNVs available for contig {contig!r}")

        debug("variant arrays")
        pos = root[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )
        coords["variant_end"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/END"], inline_array=inline_array, chunks=chunks
            ),
        )
        coords["variant_id"] = (
            [DIM_VARIANT],
            da_from_zarr(
                root[f"{contig}/variants/ID"], inline_array=inline_array, chunks=chunks
            ),
        )
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )
        for field in "Region", "StartBreakpointMethod", "EndBreakpointMethod":
            data_vars[f"variant_{field}"] = (
                [DIM_VARIANT],
                da_from_zarr(
                    root[f"{contig}/variants/{field}"],
                    inline_array=inline_array,
                    chunks=chunks,
                ),
            )

        debug("call arrays")
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE],
            da_from_zarr(
                root[f"{contig}/calldata/GT"], inline_array=inline_array, chunks=chunks
            ),
        )

        debug("sample arrays")
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )
        for field in "sample_coverage_variance", "sample_is_high_variance":
            data_vars[field] = (
                [DIM_SAMPLE],
                da_from_zarr(root[field], inline_array=inline_array, chunks=chunks),
            )

        debug("set up attributes")
        attrs = {"contigs": self.contigs}

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def cnv_discordant_read_calls(
        self,
        contig,
        sample_sets=None,
        inline_array=True,
        chunks="native",
    ):
        """Access CNV discordant read calls data.

        Parameters
        ----------
        contig : str or list of str
            Chromosome arm, e.g., "3R". Multiple values can be provided
            as a list, in which case data will be concatenated, e.g., ["2R",
            "3R"].
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of CNV alleles and genotypes.

        """
        debug = self._log.debug

        # N.B., we cannot support region instead of contig here, because some
        # CNV alleles have unknown start or end coordinates.

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        if isinstance(contig, str):
            contig = [contig]

        debug("access data and concatenate as needed")
        lx = []
        for c in contig:

            ly = []
            for s in sample_sets:
                y = self._cnv_discordant_read_calls_dataset(
                    contig=c,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            x = xarray_concat(ly, dim=DIM_SAMPLE)
            lx.append(x)

        ds = xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def gene_cnv(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
    ):
        """Compute modal copy number by gene, from HMM data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or
            a release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of modal copy number per gene and associated data.

        """

        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        ds = xarray_concat(
            [
                self._gene_cnv(
                    region=r,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    max_coverage_variance=max_coverage_variance,
                )
                for r in region
            ],
            dim="genes",
        )

        return ds

    def _gene_cnv(self, *, region, sample_sets, sample_query, max_coverage_variance):
        debug = self._log.debug

        debug("sanity check")
        assert isinstance(region, Region)

        debug("access HMM data")
        ds_hmm = self.cnv_hmm(
            region=region.contig,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )
        pos = ds_hmm["variant_position"].data
        end = ds_hmm["variant_end"].data
        cn = ds_hmm["call_CN"].data
        with self._dask_progress(desc="Load CNV HMM data"):
            pos, end, cn = dask.compute(pos, end, cn)

        debug("access genes")
        df_genome_features = self.genome_features(region=region)
        df_genes = df_genome_features.query("type == 'gene'")

        debug("setup intermediates")
        windows = []
        modes = []
        counts = []

        debug("iterate over genes")
        genes_iterator = self._progress(
            df_genes.itertuples(),
            desc="Compute modal gene copy number",
            total=len(df_genes),
        )
        for gene in genes_iterator:

            # locate windows overlapping the gene
            loc_gene_start = bisect_left(end, gene.start)
            loc_gene_stop = bisect_right(pos, gene.end)
            w = loc_gene_stop - loc_gene_start
            windows.append(w)

            # slice out copy number data for the given gene
            cn_gene = cn[loc_gene_start:loc_gene_stop]

            # compute the modes
            m, c = _cn_mode(cn_gene, vmax=12)
            modes.append(m)
            counts.append(c)

        debug("combine results")
        windows = np.array(windows)
        modes = np.vstack(modes)
        counts = np.vstack(counts)

        debug("build dataset")
        ds_out = xr.Dataset(
            coords={
                "gene_id": (["genes"], df_genes["ID"].values),
                "sample_id": (["samples"], ds_hmm["sample_id"].values),
            },
            data_vars={
                "gene_contig": (["genes"], df_genes["contig"].values),
                "gene_start": (["genes"], df_genes["start"].values),
                "gene_end": (["genes"], df_genes["end"].values),
                "gene_windows": (["genes"], windows),
                "gene_name": (["genes"], df_genes["Name"].values),
                "gene_strand": (["genes"], df_genes["strand"].values),
                "gene_description": (["genes"], df_genes["description"].values),
                "CN_mode": (["genes", "samples"], modes),
                "CN_mode_count": (["genes", "samples"], counts),
                "sample_coverage_variance": (
                    ["samples"],
                    ds_hmm["sample_coverage_variance"].values,
                ),
                "sample_is_high_variance": (
                    ["samples"],
                    ds_hmm["sample_is_high_variance"].values,
                ),
            },
        )

        return ds_out

    def gene_cnv_frequencies(
        self,
        region,
        cohorts,
        sample_query=None,
        min_cohort_size=10,
        sample_sets=None,
        drop_invariant=True,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
    ):
        """Compute modal copy number by gene, then compute the frequency of
        amplifications and deletions in one or more cohorts, from HMM data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int
            Minimum cohort size, below which cohorts are dropped.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, drop any rows where there is no evidence of variation.
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of CNV amplification (amp) and deletion (del)
            frequencies in the specified cohorts, one row per gene and CNV type
            (amp/del).

        """
        debug = self._log.debug

        debug("check and normalise parameters")
        _check_param_min_cohort_size(min_cohort_size)
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("access and concatenate data from regions")
        df = pd.concat(
            [
                self._gene_cnv_frequencies(
                    region=r,
                    cohorts=cohorts,
                    sample_query=sample_query,
                    min_cohort_size=min_cohort_size,
                    sample_sets=sample_sets,
                    drop_invariant=drop_invariant,
                    max_coverage_variance=max_coverage_variance,
                )
                for r in region
            ],
            axis=0,
        )

        debug("add metadata")
        title = f"Gene CNV frequencies ({_region_str(region)})"
        df.attrs["title"] = title

        return df

    def _gene_cnv_frequencies(
        self,
        *,
        region,
        cohorts,
        sample_query,
        min_cohort_size,
        sample_sets,
        drop_invariant,
        max_coverage_variance,
    ):
        debug = self._log.debug

        debug("sanity check - this function is one region at a time")
        assert isinstance(region, Region)

        debug("get gene copy number data")
        ds_cnv = self.gene_cnv(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )

        debug("load sample metadata")
        df_samples = self.sample_metadata(sample_sets=sample_sets)

        debug("align sample metadata with samples in CNV data")
        sample_id = ds_cnv["sample_id"].values
        df_samples = df_samples.set_index("sample_id").loc[sample_id].reset_index()

        debug("figure out expected copy number")
        if region.contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        debug(
            "setup output dataframe - two rows for each gene, one for amplification and one for deletion"
        )
        n_genes = ds_cnv.dims["genes"]
        df_genes = ds_cnv[
            [
                "gene_id",
                "gene_name",
                "gene_strand",
                "gene_description",
                "gene_contig",
                "gene_start",
                "gene_end",
            ]
        ].to_dataframe()
        df = pd.concat([df_genes, df_genes], axis=0).reset_index(drop=True)
        df.rename(
            columns={
                "gene_contig": "contig",
                "gene_start": "start",
                "gene_end": "end",
            },
            inplace=True,
        )

        debug("add CNV type column")
        df_cnv_type = pd.DataFrame(
            {
                "cnv_type": np.array(
                    (["amp"] * n_genes) + (["del"] * n_genes), dtype=object
                )
            }
        )
        df = pd.concat([df, df_cnv_type], axis=1)

        debug("set up intermediates")
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)
        is_called = cn >= 0

        debug("set up cohort dict")
        coh_dict = _locate_cohorts(cohorts=cohorts, df_samples=df_samples)

        debug("compute cohort frequencies")
        freq_cols = dict()
        for coh, loc_coh in coh_dict.items():

            n_samples = np.count_nonzero(loc_coh)
            debug(f"{coh}, {n_samples} samples")

            if n_samples >= min_cohort_size:

                # subset data to cohort
                is_amp_coh = np.compress(loc_coh, is_amp, axis=1)
                is_del_coh = np.compress(loc_coh, is_del, axis=1)
                is_called_coh = np.compress(loc_coh, is_called, axis=1)

                # count amplifications and deletions
                amp_count_coh = np.sum(is_amp_coh, axis=1)
                del_count_coh = np.sum(is_del_coh, axis=1)
                called_count_coh = np.sum(is_called_coh, axis=1)

                # compute frequencies, taking accessibility into account
                with np.errstate(divide="ignore", invalid="ignore"):
                    amp_freq_coh = np.where(
                        called_count_coh > 0, amp_count_coh / called_count_coh, np.nan
                    )
                    del_freq_coh = np.where(
                        called_count_coh > 0, del_count_coh / called_count_coh, np.nan
                    )

                freq_cols[f"frq_{coh}"] = np.concatenate([amp_freq_coh, del_freq_coh])

        debug("build a dataframe with the frequency columns")
        df_freqs = pd.DataFrame(freq_cols)

        debug("compute max_af and additional columns")
        df_extras = pd.DataFrame(
            {
                "max_af": df_freqs.max(axis=1),
                "windows": np.concatenate(
                    [ds_cnv["gene_windows"].values, ds_cnv["gene_windows"].values]
                ),
            }
        )

        debug("build the final dataframe")
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df_freqs, df_extras], axis=1)
        df.sort_values(["contig", "start", "cnv_type"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        debug("add label")
        df["label"] = _pandas_apply(
            _make_gene_cnv_label, df, columns=["gene_id", "gene_name", "cnv_type"]
        )

        debug("deal with invariants")
        if drop_invariant:
            df = df.query("max_af > 0")

        debug("set index for convenience")
        df.set_index(["gene_id", "gene_name", "cnv_type"], inplace=True)

        return df

    def open_haplotypes(self, sample_set, analysis):
        """Open haplotypes zarr.

        Parameters
        ----------
        sample_set : str
            Sample set identifier, e.g., "AG1000G-AO".
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_haplotypes[(sample_set, analysis)]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_haplotypes/{sample_set}/{analysis}/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            # some sample sets have no data for a given analysis, handle this
            try:
                root = zarr.open_consolidated(store=store)
            except FileNotFoundError:
                root = None
            self._cache_haplotypes[(sample_set, analysis)] = root
        return root

    def open_haplotype_sites(self, analysis):
        """Open haplotype sites zarr.

        Parameters
        ----------
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis,
            the "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.

        Returns
        -------
        root : zarr.hierarchy.Group

        """
        try:
            return self._cache_haplotype_sites[analysis]
        except KeyError:
            path = f"{self._base_path}/v3/snp_haplotypes/sites/{analysis}/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_haplotype_sites[analysis] = root
        return root

    def _haplotypes_dataset(
        self, *, contig, sample_set, analysis, inline_array, chunks
    ):
        debug = self._log.debug

        debug("open zarr")
        root = self.open_haplotypes(sample_set=sample_set, analysis=analysis)
        sites = self.open_haplotype_sites(analysis=analysis)

        # some sample sets have no data for a given analysis, handle this
        # TODO consider returning a dataset with 0 length samples dimension instead, would
        # probably simplify a lot of other logic
        if root is None:
            return None

        coords = dict()
        data_vars = dict()

        debug("variant_position")
        pos = sites[f"{contig}/variants/POS"]
        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )

        debug("variant_contig")
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )

        debug("variant_allele")
        ref = da_from_zarr(
            sites[f"{contig}/variants/REF"], inline_array=inline_array, chunks=chunks
        )
        alt = da_from_zarr(
            sites[f"{contig}/variants/ALT"], inline_array=inline_array, chunks=chunks
        )
        variant_allele = da.hstack([ref[:, None], alt[:, None]])
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        debug("call_genotype")
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            da_from_zarr(
                root[f"{contig}/calldata/GT"], inline_array=inline_array, chunks=chunks
            ),
        )

        debug("sample arrays")
        coords["sample_id"] = (
            [DIM_SAMPLE],
            da_from_zarr(root["samples"], inline_array=inline_array, chunks=chunks),
        )

        debug("set up attributes")
        attrs = {"contigs": self.contigs}

        debug("create a dataset")
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    def haplotypes(
        self,
        region,
        analysis,
        sample_sets=None,
        sample_query=None,
        inline_array=True,
        chunks="native",
        cohort_size=None,
        random_seed=42,
    ):
        """Access haplotype data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr
            chunks. Also, can be a target size, e.g., '200 MiB'.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of haplotypes and associated data.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)
        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        debug("build dataset")
        lx = []
        for r in region:
            ly = []

            for s in sample_sets:
                y = self._haplotypes_dataset(
                    contig=r.contig,
                    sample_set=s,
                    analysis=analysis,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                if y is not None:
                    ly.append(y)

            if len(ly) == 0:
                debug("early out, no data for given sample sets and analysis")
                return None

            debug("concatenate data from multiple sample sets")
            x = xarray_concat(ly, dim=DIM_SAMPLE)

            debug("handle region")
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        debug("handle sample query")
        if sample_query is not None:

            debug("load sample metadata")
            df_samples = self.sample_metadata(sample_sets=sample_sets)

            debug("align sample metadata with haplotypes")
            phased_samples = ds["sample_id"].values.tolist()
            df_samples_phased = (
                df_samples.set_index("sample_id").loc[phased_samples].reset_index()
            )

            debug("apply the query")
            loc_samples = df_samples_phased.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)

        debug("handle cohort size")
        if cohort_size is not None:
            n_samples = ds.dims["samples"]
            if n_samples < cohort_size:
                raise ValueError(
                    f"not enough samples ({n_samples}) for cohort size ({cohort_size})"
                )
            rng = np.random.default_rng(seed=random_seed)
            loc_downsample = rng.choice(n_samples, size=cohort_size, replace=False)
            loc_downsample.sort()
            ds = ds.isel(samples=loc_downsample)

        return ds

    def _read_cohort_metadata(self, *, sample_set):
        """Read cohort metadata for a single sample set."""
        try:
            df = self._cache_cohort_metadata[sample_set]
        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = _release_to_path(release)
            path_prefix = f"{self._base_path}/{release_path}/metadata"
            path = f"{path_prefix}/cohorts_{self._cohorts_analysis}/{sample_set}/samples.cohorts.csv"
            # N.B., not all cohort metadata files exist, need to handle FileNotFoundError
            try:
                with self._fs.open(path) as f:
                    df = pd.read_csv(f, na_values="")

                # ensure all column names are lower case
                df.columns = [c.lower() for c in df.columns]

                # rename some columns for consistent naming
                df.rename(
                    columns={
                        "adm1_iso": "admin1_iso",
                        "adm1_name": "admin1_name",
                        "adm2_name": "admin2_name",
                    },
                    inplace=True,
                )
            except FileNotFoundError:
                # Specify cohort_cols
                cohort_cols = (
                    "country_iso",
                    "admin1_name",
                    "admin1_iso",
                    "admin2_name",
                    "taxon",
                    "cohort_admin1_year",
                    "cohort_admin1_month",
                    "cohort_admin2_year",
                    "cohort_admin2_month",
                )

                # Get sample ids as an index via general metadata (has caching)
                df_general = self._read_general_metadata(sample_set=sample_set)
                df_general.set_index("sample_id", inplace=True)

                # Create a blank DataFrame with cohort_cols and sample_id index
                df = pd.DataFrame(columns=cohort_cols, index=df_general.index.copy())

                # Revert sample_id index to column
                df.reset_index(inplace=True)

            self._cache_cohort_metadata[sample_set] = df
        return df.copy()

    def sample_cohorts(self, sample_sets=None):
        """Access cohorts metadata for one or more sample sets.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of cohort metadata, one row per sample.

        """
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        # concatenate multiple sample sets
        dfs = [self._read_cohort_metadata(sample_set=s) for s in sample_sets]
        df = pd.concat(dfs, axis=0, ignore_index=True)

        return df

    def aa_allele_frequencies(
        self,
        transcript,
        cohorts,
        sample_query=None,
        min_cohort_size=10,
        site_mask=None,
        sample_sets=None,
        drop_invariant=True,
    ):
        """Compute per amino acid allele frequencies for a gene transcript.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RA".
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            `{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}`.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int
            Minimum cohort size, below which allele frequencies are not
            calculated for cohorts.
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are
            dropped from the result.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of amino acid allele frequencies, one row per
            replacement.

        Notes
        -----
        Cohorts with fewer samples than min_cohort_size will be excluded from
        output.

        """
        debug = self._log.debug

        df_snps = self.snp_allele_frequencies(
            transcript=transcript,
            cohorts=cohorts,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            sample_sets=sample_sets,
            drop_invariant=drop_invariant,
            effects=True,
        )
        df_snps.reset_index(inplace=True)

        # we just want aa change
        df_ns_snps = df_snps.query(AA_CHANGE_QUERY).copy()

        # N.B., we need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        debug("group and sum to collapse multi variant allele changes")
        freq_cols = [col for col in df_ns_snps if col.startswith("frq")]
        agg = {c: np.nansum for c in freq_cols}
        keep_cols = (
            "contig",
            "transcript",
            "aa_pos",
            "ref_allele",
            "ref_aa",
            "alt_aa",
            "effect",
            "impact",
        )
        for c in keep_cols:
            agg[c] = "first"
        agg["alt_allele"] = lambda v: "{" + ",".join(v) + "}" if len(v) > 1 else v
        df_aaf = df_ns_snps.groupby(["position", "aa_change"]).agg(agg).reset_index()

        debug("compute new max_af")
        df_aaf["max_af"] = df_aaf[freq_cols].max(axis=1)

        debug("add label")
        df_aaf["label"] = _pandas_apply(
            _make_snp_label_aa,
            df_aaf,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )

        debug("sort by genomic position")
        df_aaf = df_aaf.sort_values(["position", "aa_change"])

        debug("set index")
        df_aaf.set_index(["aa_change", "contig", "position"], inplace=True)

        debug("add metadata")
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_aaf.attrs["title"] = title

        return df_aaf

    def plot_frequencies_heatmap(
        self,
        df,
        index="label",
        max_len=100,
        x_label="Cohorts",
        y_label="Variants",
        colorbar=True,
        col_width=40,
        width=None,
        row_height=20,
        height=None,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Reds",
        title=True,
        **kwargs,
    ):
        """Plot a heatmap from a pandas DataFrame of frequencies, e.g., output
        from `Ag3.snp_allele_frequencies()` or `Ag3.gene_cnv_frequencies()`.
        It's recommended to filter the input DataFrame to just rows of interest,
        i.e., fewer rows than `max_len`.

        Parameters
        ----------
        df : pandas DataFrame
           A DataFrame of frequencies, e.g., output from
           `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
        index : str or list of str
            One or more column headers that are present in the input dataframe.
            This becomes the heatmap y-axis row labels. The column/s must
            produce a unique index.
        max_len : int, optional
            Displaying large styled dataframes may cause ipython notebooks to
            crash.
        x_label : str, optional
            This is the x-axis label that will be displayed on the heatmap.
        y_label : str, optional
            This is the y-axis label that will be displayed on the heatmap.
        colorbar : bool, optional
            If False, colorbar is not output.
        col_width : int, optional
            Plot width per column in pixels (px).
        width : int, optional
            Plot width in pixels (px), overrides col_width.
        row_height : int, optional
            Plot height per row in pixels (px).
        height : int, optional
            Plot height in pixels (px), overrides row_height.
        text_auto : str, optional
            Formatting for frequency values.
        aspect : str, optional
            Control the aspect ratio of the heatmap.
        color_continuous_scale : str, optional
            Color scale to use.
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        **kwargs
            Other parameters are passed through to px.imshow().

        Returns
        -------
        fig : plotly.graph_objects.Figure

        """
        debug = self._log.debug

        import plotly.express as px

        debug("check len of input")
        if len(df) > max_len:
            raise ValueError(f"Input DataFrame is longer than {max_len}")

        debug("handle title")
        if title is True:
            title = df.attrs.get("title", None)

        debug("indexing")
        if index is None:
            index = list(df.index.names)
        df = df.reset_index().copy()
        if isinstance(index, list):
            index_col = (
                df[index]
                .astype(str)
                .apply(
                    lambda row: ", ".join([o for o in row if o is not None]),
                    axis="columns",
                )
            )
        elif isinstance(index, str):
            index_col = df[index].astype(str)
        else:
            raise TypeError("wrong type for index parameter, expected list or str")

        debug("check that index is unique")
        if not index_col.is_unique:
            raise ValueError(f"{index} does not produce a unique index")

        debug("drop and re-order columns")
        frq_cols = [col for col in df.columns if col.startswith("frq_")]

        debug("keep only freq cols")
        heatmap_df = df[frq_cols].copy()

        debug("set index")
        heatmap_df.set_index(index_col, inplace=True)

        debug("clean column names")
        heatmap_df.columns = heatmap_df.columns.str.lstrip("frq_")

        debug("deal with width and height")
        if width is None:
            width = 400 + col_width * len(heatmap_df.columns)
            if colorbar:
                width += 40
        if height is None:
            height = 200 + row_height * len(heatmap_df)
            if title is not None:
                height += 40

        debug("plotly heatmap styling")
        fig = px.imshow(
            img=heatmap_df,
            zmin=0,
            zmax=1,
            width=width,
            height=height,
            text_auto=text_auto,
            aspect=aspect,
            color_continuous_scale=color_continuous_scale,
            title=title,
            **kwargs,
        )

        fig.update_xaxes(side="bottom", tickangle=30)
        if x_label is not None:
            fig.update_xaxes(title=x_label)
        if y_label is not None:
            fig.update_yaxes(title=y_label)
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Frequency",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            )
        )
        if not colorbar:
            fig.update(layout_coloraxis_showscale=False)

        return fig

    def snp_allele_frequencies_advanced(
        self,
        transcript,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        drop_invariant=True,
        variant_query=None,
        site_mask=None,
        nobs_mode="called",  # or "fixed"
        ci_method="wilson",
    ):
        """Group samples by taxon, area (space) and period (time), then compute
        SNP allele counts and frequencies.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RD".
        area_by : str
            Column name in the sample metadata to use to group samples
            spatially. E.g., use "admin1_iso" or "admin1_name" to group by level
            1 administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        drop_invariant : bool, optional
            If True, variants with no alternate allele calls in any cohorts are
            dropped from the result.
        variant_query : str, optional
        site_mask : str, optional
            Site filters mask to apply.
        nobs_mode : {"called", "fixed"}
            Method for calculating the denominator when computing frequencies.
            If "called" then use the number of called alleles, i.e., number of
            samples with non-missing genotype calls multiplied by 2. If "fixed"
            then use the number of samples multiplied by 2.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are
            1-dimensional arrays with data about the variants, such as the
            contig, position, reference and alternate alleles. Variables
            prefixed with "event" are 2-dimensional arrays with the allele
            counts and frequency calculations.

        """
        debug = self._log.debug

        debug("check parameters")
        _check_param_min_cohort_size(min_cohort_size)

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("access SNP calls")
        ds_snps = self.snp_calls(
            region=transcript,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )

        debug("access genotypes")
        gt = ds_snps["call_genotype"].data

        debug("prepare sample metadata for cohort grouping")
        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        debug("group samples to make cohorts")
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        debug("build cohorts dataframe")
        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort, min_cohort_size
        )

        debug("bring genotypes into memory")
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        debug("set up variant variables")
        contigs = ds_snps.attrs["contigs"]
        variant_contig = np.repeat(
            [contigs[i] for i in ds_snps["variant_contig"].values], 3
        )
        variant_position = np.repeat(ds_snps["variant_position"].values, 3)
        alleles = ds_snps["variant_allele"].values
        variant_ref_allele = np.repeat(alleles[:, 0], 3)
        variant_alt_allele = alleles[:, 1:].flatten()
        variant_pass_gamb_colu_arab = np.repeat(
            ds_snps["variant_filter_pass_gamb_colu_arab"].values, 3
        )
        variant_pass_gamb_colu = np.repeat(
            ds_snps["variant_filter_pass_gamb_colu"].values, 3
        )
        variant_pass_arab = np.repeat(ds_snps["variant_filter_pass_arab"].values, 3)

        debug("setup main event variables")
        n_variants, n_cohorts = len(variant_position), len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        debug("build event count and nobs for each cohort")
        cohorts_iterator = self._progress(
            enumerate(df_cohorts.itertuples()),
            total=len(df_cohorts),
            desc="Compute SNP allele frequencies",
        )
        for cohort_index, cohort in cohorts_iterator:

            cohort_key = cohort.taxon, cohort.area, cohort.period
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            cohort_ac, cohort_an = _cohort_alt_allele_counts_melt(
                gt, sample_indices, max_allele=3
            )
            count[:, cohort_index] = cohort_ac

            if nobs_mode == "called":
                nobs[:, cohort_index] = cohort_an
            elif nobs_mode == "fixed":
                nobs[:, cohort_index] = cohort.size * 2
            else:
                raise ValueError(f"Bad nobs_mode: {nobs_mode!r}")

        debug("compute frequency")
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = count / nobs

        debug("compute maximum frequency over cohorts")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)

        debug("make dataframe of SNPs")
        df_variants = pd.DataFrame(
            {
                "contig": variant_contig,
                "position": variant_position,
                "ref_allele": variant_ref_allele.astype("U1"),
                "alt_allele": variant_alt_allele.astype("U1"),
                "max_af": max_af,
                "pass_gamb_colu_arab": variant_pass_gamb_colu_arab,
                "pass_gamb_colu": variant_pass_gamb_colu,
                "pass_arab": variant_pass_arab,
            }
        )

        debug("deal with SNP alleles not observed")
        if drop_invariant:
            loc_variant = max_af > 0
            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)
            count = np.compress(loc_variant, count, axis=0)
            nobs = np.compress(loc_variant, nobs, axis=0)
            frequency = np.compress(loc_variant, frequency, axis=0)

        debug("set up variant effect annotator")
        ann = self._annotator()

        debug("add effects to the dataframe")
        ann.get_effects(
            transcript=transcript, variants=df_variants, progress=self._progress
        )

        debug("add variant labels")
        df_variants["label"] = _pandas_apply(
            _make_snp_label_effect,
            df_variants,
            columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
        )

        debug("build the output dataset")
        ds_out = xr.Dataset()

        debug("cohort variables")
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        debug("variant variables")
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        debug("event variables")
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        debug("apply variant query")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        debug("add confidence intervals")
        _add_frequency_ci(ds_out, ci_method)

        debug("tidy up display by sorting variables")
        ds_out = ds_out[sorted(ds_out)]

        debug("add metadata")
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_out.attrs["title"] = title

        return ds_out

    def aa_allele_frequencies_advanced(
        self,
        transcript,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        site_mask=None,
        nobs_mode="called",  # or "fixed"
        ci_method="wilson",
    ):
        """Group samples by taxon, area (space) and period (time), then compute
        amino acid change allele counts and frequencies.

        Parameters
        ----------
        transcript : str
            Gene transcript ID (AgamP4.12), e.g., "AGAP004707-RD".
        area_by : str
            Column name in the sample metadata to use to group samples spatially.
            E.g., use "admin1_iso" or "admin1_name" to group by level 1
            administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        variant_query : str, optional
        site_mask : str, optional
            Site filters mask to apply.
        nobs_mode : {"called", "fixed"}
            Method for calculating the denominator when computing frequencies.
            If "called" then use the number of called alleles, i.e., number of
            samples with non-missing genotype calls multiplied by 2. If "fixed"
            then use the number of samples multiplied by 2.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are 1-dimensional
            arrays with data about the variants, such as the contig, position,
            reference and alternate alleles. Variables prefixed with "event" are
            2-dimensional arrays with the allele counts and frequency
            calculations.

        """
        debug = self._log.debug

        debug("begin by computing SNP allele frequencies")
        ds_snp_frq = self.snp_allele_frequencies_advanced(
            transcript=transcript,
            area_by=area_by,
            period_by=period_by,
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            drop_invariant=True,  # always drop invariant for aa frequencies
            variant_query=AA_CHANGE_QUERY,  # we'll also apply a variant query later
            site_mask=site_mask,
            nobs_mode=nobs_mode,
            ci_method=None,  # we will recompute confidence intervals later
        )

        # N.B., we need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        # add in a special grouping column to work around the fact that xarray currently
        # doesn't support grouping by multiple variables in the same dimension
        df_grouper = ds_snp_frq[
            ["variant_position", "variant_aa_change"]
        ].to_dataframe()
        grouper_var = df_grouper.apply(
            lambda row: "_".join([str(v) for v in row]), axis="columns"
        )
        ds_snp_frq["variant_position_aa_change"] = "variants", grouper_var

        debug("group by position and amino acid change")
        group_by_aa_change = ds_snp_frq.groupby("variant_position_aa_change")

        debug("apply aggregation")
        ds_aa_frq = group_by_aa_change.map(_map_snp_to_aa_change_frq_ds)

        debug("add back in cohort variables, unaffected by aggregation")
        cohort_vars = [v for v in ds_snp_frq if v.startswith("cohort_")]
        for v in cohort_vars:
            ds_aa_frq[v] = ds_snp_frq[v]

        debug("sort by genomic position")
        ds_aa_frq = ds_aa_frq.sortby(["variant_position", "variant_aa_change"])

        debug("recompute frequency")
        count = ds_aa_frq["event_count"].values
        nobs = ds_aa_frq["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = count / nobs  # ignore division warnings
        ds_aa_frq["event_frequency"] = ("variants", "cohorts"), frequency

        debug("recompute max frequency over cohorts")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(ds_aa_frq["event_frequency"].values, axis=1)
        ds_aa_frq["variant_max_af"] = "variants", max_af

        debug("set up variant dataframe, useful intermediate")
        variant_cols = [v for v in ds_aa_frq if v.startswith("variant_")]
        df_variants = ds_aa_frq[variant_cols].to_dataframe()
        df_variants.columns = [c.split("variant_")[1] for c in df_variants.columns]

        debug("assign new variant label")
        label = _pandas_apply(
            _make_snp_label_aa,
            df_variants,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )
        ds_aa_frq["variant_label"] = "variants", label

        debug("apply variant query if given")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_aa_frq = ds_aa_frq.isel(variants=loc_variants)

        debug("compute new confidence intervals")
        _add_frequency_ci(ds_aa_frq, ci_method)

        debug("tidy up display by sorting variables")
        ds_aa_frq = ds_aa_frq[sorted(ds_aa_frq)]

        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_aa_frq.attrs["title"] = title

        return ds_aa_frq

    def gene_cnv_frequencies_advanced(
        self,
        region,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=True,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        ci_method="wilson",
    ):
        """Group samples by taxon, area (space) and period (time), then compute
        gene CNV counts and frequencies.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        area_by : str
            Column name in the sample metadata to use to group samples spatially.
            E.g., use "admin1_iso" or "admin1_name" to group by level 1
            administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        variant_query : str, optional
        drop_invariant : bool, optional
            If True, drop any rows where there is no evidence of variation.
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are 1-dimensional
            arrays with data about the variants, such as the contig, position,
            reference and alternate alleles. Variables prefixed with "event" are
            2-dimensional arrays with the allele counts and frequency
            calculations.

        """

        _check_param_min_cohort_size(min_cohort_size)

        region = self.resolve_region(region)
        if isinstance(region, Region):
            region = [region]

        ds = xarray_concat(
            [
                self._gene_cnv_frequencies_advanced(
                    region=r,
                    area_by=area_by,
                    period_by=period_by,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    min_cohort_size=min_cohort_size,
                    variant_query=variant_query,
                    drop_invariant=drop_invariant,
                    max_coverage_variance=max_coverage_variance,
                    ci_method=ci_method,
                )
                for r in region
            ],
            dim="variants",
        )

        title = f"Gene CNV frequencies ({_region_str(region)})"
        ds.attrs["title"] = title

        return ds

    def _gene_cnv_frequencies_advanced(
        self,
        *,
        region,
        area_by,
        period_by,
        sample_sets,
        sample_query,
        min_cohort_size,
        variant_query,
        drop_invariant,
        max_coverage_variance,
        ci_method,
    ):
        debug = self._log.debug

        debug("sanity check - here we deal with one region only")
        assert isinstance(region, Region)

        debug("access gene CNV calls")
        ds_cnv = self.gene_cnv(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )

        debug("load sample metadata")
        df_samples = self.sample_metadata(sample_sets=sample_sets)

        debug("align sample metadata")
        sample_id = ds_cnv["sample_id"].values
        df_samples = df_samples.set_index("sample_id").loc[sample_id].reset_index()

        debug("prepare sample metadata for cohort grouping")
        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        debug("group samples to make cohorts")
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        debug("build cohorts dataframe")
        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort, min_cohort_size
        )

        debug("figure out expected copy number")
        if region.contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        debug("set up intermediates")
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)
        is_called = cn >= 0

        debug("set up main event variables")
        n_genes = ds_cnv.dims["genes"]
        n_variants, n_cohorts = n_genes * 2, len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        debug("build event count and nobs for each cohort")
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):

            # construct grouping key
            cohort_key = cohort.taxon, cohort.area, cohort.period

            # obtain sample indices for cohort
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            # select genotype data for cohort
            cohort_is_amp = np.take(is_amp, sample_indices, axis=1)
            cohort_is_del = np.take(is_del, sample_indices, axis=1)
            cohort_is_called = np.take(is_called, sample_indices, axis=1)

            # compute cohort allele counts
            np.sum(cohort_is_amp, axis=1, out=count[::2, cohort_index])
            np.sum(cohort_is_del, axis=1, out=count[1::2, cohort_index])

            # compute cohort allele numbers
            cohort_n_called = np.sum(cohort_is_called, axis=1)
            nobs[:, cohort_index] = np.repeat(cohort_n_called, 2)

        debug("compute frequency")
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = np.where(nobs > 0, count / nobs, np.nan)

        debug("make dataframe of variants")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)
        df_variants = pd.DataFrame(
            {
                "contig": region.contig,
                "start": np.repeat(ds_cnv["gene_start"].values, 2),
                "end": np.repeat(ds_cnv["gene_end"].values, 2),
                "windows": np.repeat(ds_cnv["gene_windows"].values, 2),
                # alternate amplification and deletion
                "cnv_type": np.tile(np.array(["amp", "del"]), n_genes),
                "max_af": max_af,
                "gene_id": np.repeat(ds_cnv["gene_id"].values, 2),
                "gene_name": np.repeat(ds_cnv["gene_name"].values, 2),
                "gene_strand": np.repeat(ds_cnv["gene_strand"].values, 2),
            }
        )

        debug("add variant label")
        df_variants["label"] = _pandas_apply(
            _make_gene_cnv_label,
            df_variants,
            columns=["gene_id", "gene_name", "cnv_type"],
        )

        debug("build the output dataset")
        ds_out = xr.Dataset()

        debug("cohort variables")
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        debug("variant variables")
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        debug("event variables")
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        debug("deal with invariants")
        if drop_invariant:
            loc_variant = df_variants["max_af"].values > 0
            ds_out = ds_out.isel(variants=loc_variant)
            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)

        debug("apply variant query")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        debug("add confidence intervals")
        _add_frequency_ci(ds_out, ci_method)

        debug("tidy up display by sorting variables")
        ds_out = ds_out[sorted(ds_out)]

        return ds_out

    def plot_frequencies_time_series(
        self, ds, height=None, width=None, title=True, **kwargs
    ):
        """Create a time series plot of variant frequencies using plotly.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by
            `Ag3.snp_allele_frequencies_advanced()`,
            `Ag3.aa_allele_frequencies_advanced()` or
            `Ag3.gene_cnv_frequencies_advanced()`.
        height : int, optional
            Height of plot in pixels (px).
        width : int, optional
            Width of plot in pixels (px).
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        **kwargs
            Passed through to `px.line()`.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            A plotly figure containing line graphs. The resulting figure will
            have one panel per cohort, grouped into columns by taxon, and
            grouped into rows by area. Markers and lines show frequencies of
            variants.

        """
        debug = self._log.debug

        import plotly.express as px

        debug("handle title")
        if title is True:
            title = ds.attrs.get("title", None)

        debug("extract cohorts into a dataframe")
        cohort_vars = [v for v in ds if v.startswith("cohort_")]
        df_cohorts = ds[cohort_vars].to_dataframe()
        df_cohorts.columns = [c.split("cohort_")[1] for c in df_cohorts.columns]

        debug("extract variant labels")
        variant_labels = ds["variant_label"].values

        debug("build a long-form dataframe from the dataset")
        dfs = []
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
            ds_cohort = ds.isel(cohorts=cohort_index)
            df = pd.DataFrame(
                {
                    "taxon": cohort.taxon,
                    "area": cohort.area,
                    "date": cohort.period_start,
                    "period": str(
                        cohort.period
                    ),  # use string representation for hover label
                    "sample_size": cohort.size,
                    "variant": variant_labels,
                    "count": ds_cohort["event_count"].values,
                    "nobs": ds_cohort["event_nobs"].values,
                    "frequency": ds_cohort["event_frequency"].values,
                    "frequency_ci_low": ds_cohort["event_frequency_ci_low"].values,
                    "frequency_ci_upp": ds_cohort["event_frequency_ci_upp"].values,
                }
            )
            dfs.append(df)
        df_events = pd.concat(dfs, axis=0).reset_index(drop=True)

        debug("remove events with no observations")
        df_events = df_events.query("nobs > 0")

        debug("calculate error bars")
        frq = df_events["frequency"]
        frq_ci_low = df_events["frequency_ci_low"]
        frq_ci_upp = df_events["frequency_ci_upp"]
        df_events["frequency_error"] = frq_ci_upp - frq
        df_events["frequency_error_minus"] = frq - frq_ci_low

        debug("make a plot")
        fig = px.line(
            df_events,
            facet_col="taxon",
            facet_row="area",
            x="date",
            y="frequency",
            error_y="frequency_error",
            error_y_minus="frequency_error_minus",
            color="variant",
            markers=True,
            hover_name="variant",
            hover_data={
                "frequency": ":.0%",
                "period": True,
                "area": True,
                "taxon": True,
                "sample_size": True,
                "date": False,
                "variant": False,
            },
            height=height,
            width=width,
            title=title,
            labels={
                "date": "Date",
                "frequency": "Frequency",
                "variant": "Variant",
                "taxon": "Taxon",
                "area": "Area",
                "period": "Period",
                "sample_size": "Sample size",
            },
            **kwargs,
        )

        debug("tidy plot")
        fig.update_layout(yaxis_range=[-0.05, 1.05])

        return fig

    def plot_frequencies_map_markers(self, m, ds, variant, taxon, period, clear=True):
        """Plot markers on a map showing variant frequencies for cohorts grouped
        by area (space), period (time) and taxon.

        Parameters
        ----------
        m : ipyleaflet.Map
            The map on which to add the markers.
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by
            `Ag3.snp_allele_frequencies_advanced()`,
            `Ag3.aa_allele_frequencies_advanced()` or
            `Ag3.gene_cnv_frequencies_advanced()`.
        variant : int or str
            Index or label of variant to plot.
        taxon : str
            Taxon to show markers for.
        period : pd.Period
            Time period to show markers for.
        clear : bool, optional
            If True, clear all layers (except the base layer) from the map
            before adding new markers.

        """
        debug = self._log.debug

        import ipyleaflet
        import ipywidgets

        debug("slice dataset to variant of interest")
        if isinstance(variant, int):
            ds_variant = ds.isel(variants=variant)
            variant_label = ds["variant_label"].values[variant]
        elif isinstance(variant, str):
            ds_variant = ds.set_index(variants="variant_label").sel(variants=variant)
            variant_label = variant
        else:
            raise TypeError(
                f"Bad type for variant parameter; expected int or str, found {type(variant)}."
            )

        debug("convert to a dataframe for convenience")
        df_markers = ds_variant[
            [
                "cohort_taxon",
                "cohort_area",
                "cohort_period",
                "cohort_lat_mean",
                "cohort_lon_mean",
                "cohort_size",
                "event_frequency",
                "event_frequency_ci_low",
                "event_frequency_ci_upp",
            ]
        ].to_dataframe()

        debug("select data matching taxon and period parameters")
        df_markers = df_markers.loc[
            (
                (df_markers["cohort_taxon"] == taxon)
                & (df_markers["cohort_period"] == period)
            )
        ]

        debug("clear existing layers in the map")
        if clear:
            for layer in m.layers[1:]:
                m.remove_layer(layer)

        debug("add markers")
        for x in df_markers.itertuples():
            marker = ipyleaflet.CircleMarker()
            marker.location = (x.cohort_lat_mean, x.cohort_lon_mean)
            marker.radius = 20
            marker.color = "black"
            marker.weight = 1
            marker.fill_color = "red"
            marker.fill_opacity = x.event_frequency
            popup_html = f"""
                <strong>{variant_label}</strong> <br/>
                Taxon: {x.cohort_taxon} <br/>
                Area: {x.cohort_area} <br/>
                Period: {x.cohort_period} <br/>
                Sample size: {x.cohort_size} <br/>
                Frequency: {x.event_frequency:.0%}
                (95% CI: {x.event_frequency_ci_low:.0%} - {x.event_frequency_ci_upp:.0%})
            """
            marker.popup = ipyleaflet.Popup(
                child=ipywidgets.HTML(popup_html),
            )
            m.add_layer(marker)

    def plot_frequencies_interactive_map(
        self,
        ds,
        center=(-2, 20),
        zoom=3,
        title=True,
        epilogue=True,
    ):
        """Create an interactive map with markers showing variant frequencies or
        cohorts grouped by area (space), period (time) and taxon.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset of variant frequencies, such as returned by
            `Ag3.snp_allele_frequencies_advanced()`,
            `Ag3.aa_allele_frequencies_advanced()` or
            `Ag3.gene_cnv_frequencies_advanced()`.
        center : tuple of int, optional
            Location to center the map.
        zoom : int, optional
            Initial zoom level.
        title : bool or str, optional
            If True, attempt to use metadata from input dataset as a plot
            title. Otherwise, use supplied value as a title.
        epilogue : bool or str, optional
            Additional text to display below the map.

        Returns
        -------
        out : ipywidgets.Widget
            An interactive map with widgets for selecting which variant, taxon
            and time period to display.

        """
        debug = self._log.debug

        import ipyleaflet
        import ipywidgets

        debug("handle title")
        if title is True:
            title = ds.attrs.get("title", None)

        debug("create a map")
        freq_map = ipyleaflet.Map(center=center, zoom=zoom)

        debug("set up interactive controls")
        variants = ds["variant_label"].values
        taxa = np.unique(ds["cohort_taxon"].values)
        periods = np.unique(ds["cohort_period"].values)
        controls = ipywidgets.interactive(
            self.plot_frequencies_map_markers,
            m=ipywidgets.fixed(freq_map),
            ds=ipywidgets.fixed(ds),
            variant=ipywidgets.Dropdown(options=variants, description="Variant: "),
            taxon=ipywidgets.Dropdown(options=taxa, description="Taxon: "),
            period=ipywidgets.Dropdown(options=periods, description="Period: "),
            clear=ipywidgets.fixed(True),
        )

        debug("lay out widgets")
        components = []
        if title is not None:
            components.append(ipywidgets.HTML(value=f"<h3>{title}</h3>"))
        components.append(controls)
        components.append(freq_map)
        if epilogue is True:
            epilogue = """
                Variant frequencies are shown as coloured markers. Opacity of color
                denotes frequency. Click on a marker for more information.
            """
        if epilogue:
            components.append(ipywidgets.HTML(value=f"{epilogue}"))

        out = ipywidgets.VBox(components)

        return out

    def resolve_region(self, region):
        """Convert a genome region into a standard data structure.

        Parameters
        ----------
        region: str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").

        Returns
        -------
        out : Region
            A named tuple with attributes contig, start and end.

        """

        return resolve_region(self, region)

    def plot_genes(
        self,
        region,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=DEFAULT_GENES_TRACK_HEIGHT,
        show=True,
        toolbar_location="above",
        x_range=None,
        title="Genes",
    ):
        """Plot a genes track, using bokeh.

        Parameters
        ----------
        region : str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        show : bool, optional
            If true, show the plot.
        toolbar_location : str, optional
            Location of bokeh toolbar.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).
        title : str, optional
            Plot title.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("handle region parameter - this determines the genome region to plot")
        region = self.resolve_region(region)
        contig = region.contig
        start = region.start
        end = region.end
        if start is None:
            start = 0
        if end is None:
            end = len(self.genome_sequence(contig))

        debug("define x axis range")
        if x_range is None:
            x_range = bkmod.Range1d(start, end, bounds="auto")

        debug("select the genes overlapping the requested region")
        df_genome_features = self.genome_features(
            attributes=["ID", "Name", "Parent", "description"]
        )
        data = df_genome_features.query(
            f"type == 'gene' and contig == '{contig}' and start < {end} and end > {start}"
        ).copy()

        debug(
            "we're going to plot each gene as a rectangle, so add some additional columns"
        )
        data["bottom"] = np.where(data["strand"] == "+", 1, 0)
        data["top"] = data["bottom"] + 0.8

        debug("tidy up some columns for presentation")
        data["Name"].fillna("", inplace=True)
        data["description"].fillna("", inplace=True)

        debug("define tooltips for hover")
        tooltips = [
            ("ID", "@ID"),
            ("Name", "@Name"),
            ("Description", "@description"),
            ("Location", "@contig:@start{,}-@end{,}"),
        ]

        debug("make a figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=title,
            plot_width=width,
            plot_height=height,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "tap",
                "hover",
            ],
            toolbar_location=toolbar_location,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            tooltips=tooltips,
            x_range=x_range,
        )

        debug("add functionality to click through to vectorbase")
        url = "https://vectorbase.org/vectorbase/app/record/gene/@ID"
        taptool = fig.select(type=bkmod.TapTool)
        taptool.callback = bkmod.OpenURL(url=url)

        debug("now plot the genes as rectangles")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data,
            line_width=0.5,
            fill_alpha=0.5,
        )

        debug("tidy up the plot")
        fig.y_range = bkmod.Range1d(-0.4, 2.2)
        fig.ygrid.visible = False
        yticks = [0.4, 1.4]
        yticklabels = ["-", "+"]
        fig.yaxis.ticker = yticks
        fig.yaxis.major_label_overrides = {k: v for k, v in zip(yticks, yticklabels)}
        _bokeh_style_genome_xaxis(fig, region.contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_transcript(
        self,
        transcript,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=DEFAULT_GENES_TRACK_HEIGHT,
        show=True,
        x_range=None,
        toolbar_location="above",
        title=True,
    ):
        """Plot a transcript, using bokeh.

        Parameters
        ----------
        transcript : str
            Transcript identifier, e.g., "AGAP004707-RD".
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        show : bool, optional
            If true, show the plot.
        toolbar_location : str, optional
            Location of bokeh toolbar.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).
        title : str, optional
            Plot title.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("find the transcript annotation")
        df_genome_features = self.genome_features().set_index("ID")
        parent = df_genome_features.loc[transcript]

        if title is True:
            title = f"{transcript} ({parent.strand})"

        debug("define tooltips for hover")
        tooltips = [
            ("Type", "@type"),
            ("Location", "@contig:@start{,}-@end{,}"),
        ]

        debug("make a figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=title,
            plot_width=width,
            plot_height=height,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset", "hover"],
            toolbar_location=toolbar_location,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            tooltips=tooltips,
            x_range=x_range,
        )

        debug("find child components of the transcript")
        data = df_genome_features.set_index("Parent").loc[transcript].copy()
        data["bottom"] = -0.4
        data["top"] = 0.4

        debug("plot exons")
        exons = data.query("type == 'exon'")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=exons,
            fill_color=None,
            line_color="black",
            line_width=0.5,
            fill_alpha=0,
        )

        debug("plot introns")
        for intron_start, intron_end in zip(exons[:-1]["end"], exons[1:]["start"]):
            intron_midpoint = (intron_start + intron_end) / 2
            line_data = pd.DataFrame(
                {
                    "x": [intron_start, intron_midpoint, intron_end],
                    "y": [0, 0.1, 0],
                    "type": "intron",
                    "contig": parent.contig,
                    "start": intron_start,
                    "end": intron_end,
                }
            )
            fig.line(
                x="x",
                y="y",
                source=line_data,
                line_width=1,
                line_color="black",
            )

        debug("plot UTRs")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'five_prime_UTR'"),
            fill_color="green",
            line_width=0,
            fill_alpha=0.5,
        )
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'three_prime_UTR'"),
            fill_color="red",
            line_width=0,
            fill_alpha=0.5,
        )

        debug("plot CDSs")
        fig.quad(
            bottom="bottom",
            top="top",
            left="start",
            right="end",
            source=data.query("type == 'CDS'"),
            fill_color="blue",
            line_width=0,
            fill_alpha=0.5,
        )

        debug("tidy up the figure")
        fig.yaxis.ticker = []
        fig.y_range = bkmod.Range1d(-0.6, 0.6)
        _bokeh_style_genome_xaxis(fig, parent.contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_cnv_hmm_coverage_track(
        self,
        sample,
        region,
        sample_set=None,
        y_max="auto",
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=200,
        circle_kwargs=None,
        line_kwargs=None,
        show=True,
        x_range=None,
    ):
        """Plot CNV HMM data for a single sample, using bokeh.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_set : str, optional
            Sample set identifier.
        y_max : str or int, optional
            Maximum Y axis value.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        line_kwargs : dict, optional
            Passed through to bokeh line() function.
        show : bool, optional
            If true, show the plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("resolve region")
        region = self.resolve_region(region)

        debug("access sample metadata, look up sample")
        sample_rec = self._lookup_sample(sample=sample, sample_set=sample_set)
        sample_id = sample_rec.name  # sample_id
        sample_set = sample_rec["sample_set"]

        debug("access HMM data")
        hmm = self.cnv_hmm(
            region=region, sample_sets=sample_set, max_coverage_variance=None
        )

        debug("select data for the given sample")
        hmm_sample = hmm.set_index(samples="sample_id").sel(samples=sample_id)

        debug("extract data into a pandas dataframe for easy plotting")
        data = hmm_sample[
            ["variant_position", "variant_end", "call_NormCov", "call_CN"]
        ].to_dataframe()

        debug("add window midpoint for plotting accuracy")
        data["variant_midpoint"] = data["variant_position"] + 150

        debug("remove data where HMM is not called")
        data = data.query("call_CN >= 0")

        debug("set up y range")
        if y_max == "auto":
            y_max = data["call_CN"].max() + 2

        debug("set up x range")
        x_min = data["variant_position"].values[0]
        x_max = data["variant_end"].values[-1]
        if x_range is None:
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        debug("create a figure for plotting")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=f"CNV HMM - {sample_id} ({sample_set})",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            plot_width=width,
            plot_height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, y_max),
        )

        debug("plot the normalised coverage data")
        if circle_kwargs is None:
            circle_kwargs = dict()
        circle_kwargs.setdefault("size", 3)
        circle_kwargs.setdefault("line_width", 0.5)
        circle_kwargs.setdefault("line_color", "black")
        circle_kwargs.setdefault("fill_color", None)
        circle_kwargs.setdefault("legend_label", "Coverage")
        fig.circle(x="variant_midpoint", y="call_NormCov", source=data, **circle_kwargs)

        debug("plot the HMM state")
        if line_kwargs is None:
            line_kwargs = dict()
        line_kwargs.setdefault("width", 2)
        line_kwargs.setdefault("legend_label", "HMM")
        fig.line(x="variant_midpoint", y="call_CN", source=data, **line_kwargs)

        debug("tidy up the plot")
        fig.yaxis.axis_label = "Copy number"
        fig.yaxis.ticker = list(range(y_max + 1))
        _bokeh_style_genome_xaxis(fig, region.contig)
        fig.add_layout(fig.legend[0], "right")

        if show:
            bkplt.show(fig)

        return fig

    def plot_cnv_hmm_coverage(
        self,
        sample,
        region,
        sample_set=None,
        y_max="auto",
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=170,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        circle_kwargs=None,
        line_kwargs=None,
        show=True,
    ):
        """Plot CNV HMM data for a single sample, together with a genes track,
        using bokeh.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_set : str, optional
            Sample set identifier.
        y_max : str or int, optional
            Maximum Y axis value.
        width : int, optional
            Plot width in pixels (px).
        track_height : int, optional
            Height of CNV HMM track in pixels (px).
        genes_height : int, optional
            Height of genes track in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        line_kwargs : dict, optional
            Passed through to bokeh line() function.
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        debug("plot the main track")
        fig1 = self.plot_cnv_hmm_coverage_track(
            sample=sample,
            sample_set=sample_set,
            region=region,
            y_max=y_max,
            width=width,
            height=track_height,
            circle_kwargs=circle_kwargs,
            line_kwargs=line_kwargs,
            show=False,
        )
        fig1.xaxis.visible = False

        debug("plot genes track")
        fig2 = self.plot_genes(
            region=region,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        debug("combine plots into a single figure")
        fig = bklay.gridplot(
            [fig1, fig2], ncols=1, toolbar_location="above", merge_tools=True
        )

        if show:
            bkplt.show(fig)

        return fig

    def plot_cnv_hmm_heatmap_track(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        row_height=7,
        height=None,
        show=True,
    ):
        """Plot CNV HMM data for multiple samples as a heatmap, using bokeh.

        Parameters
        ----------
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.
        width : int, optional
            Plot width in pixels (px).
        row_height : int, optional
            Plot height per row (sample) in pixels (px).
        height : int, optional
            Absolute plot height in pixels (px), overrides row_height.
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.palettes as bkpal
        import bokeh.plotting as bkplt

        region = self.resolve_region(region)

        debug("access HMM data")
        ds_cnv = self.cnv_hmm(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )

        debug("access copy number data")
        cn = ds_cnv["call_CN"].values
        ncov = ds_cnv["call_NormCov"].values
        start = ds_cnv["variant_position"].values
        end = ds_cnv["variant_end"].values
        n_windows, n_samples = cn.shape

        debug("figure out X axis limits from data")
        x_min = start[0]
        x_max = end[-1]

        debug("set up plot title")
        title = "CNV HMM"
        if sample_sets is not None:
            if isinstance(sample_sets, (list, tuple)):
                sample_sets_text = ", ".join(sample_sets)
            else:
                sample_sets_text = sample_sets
            title += f" - {sample_sets_text}"
        if sample_query is not None:
            title += f" ({sample_query})"

        debug("figure out plot height")
        if height is None:
            plot_height = 100 + row_height * n_samples
        else:
            plot_height = height

        debug("set up figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        tooltips = [
            ("Position", "$x{0,0}"),
            ("Sample ID", "@sample_id"),
            ("HMM state", "@hmm_state"),
            ("Normalised coverage", "@norm_cov"),
        ]
        fig = bkplt.figure(
            title=title,
            plot_width=width,
            plot_height=plot_height,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            toolbar_location="above",
            x_range=bkmod.Range1d(x_min, x_max, bounds="auto"),
            y_range=(-0.5, n_samples - 0.5),
            tooltips=tooltips,
        )

        debug("set up palette and color mapping")
        palette = ("#cccccc",) + bkpal.PuOr5
        color_mapper = bkmod.LinearColorMapper(low=-1.5, high=4.5, palette=palette)

        debug("plot the HMM copy number data as an image")
        sample_id = ds_cnv["sample_id"].values
        sample_id_tiled = np.broadcast_to(sample_id[np.newaxis, :], cn.shape)
        data = dict(
            hmm_state=[cn.T],
            norm_cov=[ncov.T],
            sample_id=[sample_id_tiled.T],
            x=[x_min],
            y=[-0.5],
            dw=[n_windows * 300],
            dh=[n_samples],
        )
        fig.image(
            source=data,
            image="hmm_state",
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            color_mapper=color_mapper,
        )

        debug("tidy")
        fig.yaxis.axis_label = "Samples"
        _bokeh_style_genome_xaxis(fig, region.contig)
        fig.yaxis.ticker = bkmod.FixedTicker(
            ticks=np.arange(len(sample_id)),
        )
        fig.yaxis.major_label_overrides = {i: s for i, s in enumerate(sample_id)}
        fig.yaxis.major_label_text_font_size = f"{row_height}px"

        debug("add color bar")
        color_bar = bkmod.ColorBar(
            title="Copy number",
            color_mapper=color_mapper,
            major_label_overrides={
                -1: "unknown",
                4: "4+",
            },
            major_label_policy=bkmod.AllLabels(),
        )
        fig.add_layout(color_bar, "right")

        if show:
            bkplt.show(fig)

        return fig

    def plot_cnv_hmm_heatmap(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        row_height=7,
        track_height=None,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        show=True,
    ):
        """Plot CNV HMM data for multiple samples as a heatmap, with a genes
        track, using bokeh.

        Parameters
        ----------
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.
        width : int, optional
            Plot width in pixels (px).
        row_height : int, optional
            Plot height per row (sample) in pixels (px).
        track_height : int, optional
            Absolute plot height for HMM track in pixels (px), overrides
            row_height.
        genes_height : int, optional
            Height of genes track in pixels (px).
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        debug("plot the main track")
        fig1 = self.plot_cnv_hmm_heatmap_track(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
            width=width,
            row_height=row_height,
            height=track_height,
            show=False,
        )
        fig1.xaxis.visible = False

        debug("plot genes track")
        fig2 = self.plot_genes(
            region=region,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        debug("combine plots into a single figure")
        fig = bklay.gridplot(
            [fig1, fig2], ncols=1, toolbar_location="above", merge_tools=True
        )

        if show:
            bkplt.show(fig)

        return fig

    def igv(self, region, tracks=None):
        """Create an IGV browser and display it within the notebook.

        Parameters
        ----------
        region: str or Region
            Genomic region defined with coordinates, e.g., "2L:2422600-2422700".
        tracks : list of dict, optional
            Configuration for any additional tracks.

        Returns
        -------
        browser : igv_notebook.Browser

        """
        debug = self._log.debug

        debug("resolve region")
        region = self.resolve_region(region)

        import igv_notebook

        config = {
            "reference": {
                "id": "AgamP4",
                "name": "Anopheles gambiae (PEST)",
                "fastaURL": f"{GCS_URL}{GENOME_FASTA_PATH}",
                "indexURL": f"{GCS_URL}{GENOME_FAI_PATH}",
                "tracks": [
                    {
                        "name": "Genes",
                        "url": f"{GCS_URL}{self._genome_feature_GFF_path}",
                        "indexed": False,
                    }
                ],
            },
            "locus": str(region),
        }
        if tracks:
            config["tracks"] = tracks

        debug(config)

        igv_notebook.init()
        browser = igv_notebook.Browser(config)

        return browser

    def wgs_data_catalog(self, sample_set):
        """Load a data catalog providing URLs for downloading BAM, VCF and Zarr
        files for samples in a given sample set.

        Parameters
        ----------
        sample_set : str
            Sample set identifier.

        Returns
        -------
        df : pandas.DataFrame
            One row per sample, columns provide URLs.

        """
        debug = self._log.debug

        debug("look up release for sample set")
        release = self._lookup_release(sample_set=sample_set)
        release_path = _release_to_path(release=release)

        if release == "3.0":

            debug("special handling for 3.0 as data catalogs have a different format")

            debug("load alignments catalog")
            alignments_path = f"{self._base_path}/{release_path}/alignments/catalog.csv"
            with self._fs.open(alignments_path) as f:
                alignments_df = pd.read_csv(f, na_values="").query(
                    f"sample_set == '{sample_set}'"
                )

            debug("load SNP genotypes catalog")
            genotypes_path = (
                f"{self._base_path}/{release_path}/snp_genotypes/per_sample/catalog.csv"
            )
            with self._fs.open(genotypes_path) as f:
                genotypes_df = pd.read_csv(f, na_values="").query(
                    f"sample_set == '{sample_set}'"
                )

            debug("join catalogs")
            df = pd.merge(
                left=alignments_df, right=genotypes_df, on="sample_id", how="inner"
            )

            debug("normalise columns")
            df = df[["sample_id", "bam_path", "vcf_path", "zarr_path"]]
            df = df.rename(
                columns={
                    "bam_path": "alignments_bam",
                    "vcf_path": "snp_genotypes_vcf",
                    "zarr_path": "snp_genotypes_zarr",
                }
            )

        else:

            debug("load data catalog")
            path = f"{self._base_path}/{release_path}/metadata/general/{sample_set}/wgs_snp_data.csv"
            with self._fs.open(path) as f:
                df = pd.read_csv(f, na_values="")

            debug("normalise columns")
            df = df[
                [
                    "sample_id",
                    "alignments_bam",
                    "snp_genotypes_vcf",
                    "snp_genotypes_zarr",
                ]
            ]

        return df

    def view_alignments(
        self,
        region,
        sample,
        visibility_window=20_000,
    ):
        """Launch IGV and view sequence read alignments and SNP genotypes from
        the given sample.

        Parameters
        ----------
        region: str or Region
            Genomic region defined with coordinates, e.g., "2L:2422600-2422700".
        sample : str
            Sample identifier, e.g., "AR0001-C".
        visibility_window : int, optional
            Zoom level in base pairs at which alignment and SNP data will become
            visible.

        Notes
        -----
        Only samples from the Ag3.0 release are currently supported.

        """
        debug = self._log.debug

        debug("look up sample set for sample")
        sample_rec = self.sample_metadata().set_index("sample_id").loc[sample]
        sample_set = sample_rec["sample_set"]

        debug("load data catalog")
        df_cat = self.wgs_data_catalog(sample_set=sample_set)

        debug("locate record for sample")
        cat_rec = df_cat.set_index("sample_id").loc[sample]
        bam_url = cat_rec["alignments_bam"]
        vcf_url = cat_rec["snp_genotypes_vcf"]
        debug(bam_url)
        debug(vcf_url)

        tracks = []

        # https://github.com/igvteam/igv-notebook/issues/3 -- resolved now
        debug("set up site filters tracks")
        region = self.resolve_region(region)
        contig = region.contig
        for site_mask in self._site_mask_ids():
            site_filters_vcf_url = f"gs://vo_agam_release/v3/site_filters/{self._site_filters_analysis}/vcf/{site_mask}/{contig}_sitefilters.vcf.gz"  # noqa
            debug(site_filters_vcf_url)
            track_config = {
                "name": f"Filters - {site_mask}",
                "url": site_filters_vcf_url,
                "indexURL": f"{site_filters_vcf_url}.tbi",
                "format": "vcf",
                "type": "variant",
                "visibilityWindow": visibility_window,  # bp
                "height": 30,
                "colorBy": "FILTER",
                "colorTable": {
                    "PASS": "#00cc96",
                    "*": "#ef553b",
                },
            }
            tracks.append(track_config)

        debug("add SNPs track")
        tracks.append(
            {
                "name": "SNPs",
                "url": vcf_url,
                "indexURL": f"{vcf_url}.tbi",
                "format": "vcf",
                "type": "variant",
                "visibilityWindow": visibility_window,  # bp
                "height": 50,
            }
        )

        debug("add alignments track")
        tracks.append(
            {
                "name": "Alignments",
                "url": bam_url,
                "indexURL": f"{bam_url}.bai",
                "format": "bam",
                "type": "alignment",
                "visibilityWindow": visibility_window,  # bp
                "height": 500,
            }
        )

        debug("create IGV browser")
        self.igv(region=region, tracks=tracks)

    def results_cache_get(self, *, name, params):
        debug = self._log.debug
        if self._results_cache is None:
            raise CacheMiss
        params = params.copy()
        params["cohorts_analysis"] = self._cohorts_analysis
        params["species_analysis"] = self._species_analysis
        params["site_filters_analysis"] = self._site_filters_analysis
        cache_key, _ = hash_params(params)
        cache_path = self._results_cache / name / cache_key
        results_path = cache_path / "results.npz"
        if not results_path.exists():
            raise CacheMiss
        results = np.load(results_path)
        debug(f"loaded {name}/{cache_key}")
        return results

    def results_cache_set(self, *, name, params, results):
        debug = self._log.debug
        if self._results_cache is None:
            debug("no results cache has been configured, do nothing")
            return
        params = params.copy()
        params["cohorts_analysis"] = self._cohorts_analysis
        params["species_analysis"] = self._species_analysis
        params["site_filters_analysis"] = self._site_filters_analysis
        cache_key, params_json = hash_params(params)
        cache_path = self._results_cache / name / cache_key
        cache_path.mkdir(exist_ok=True, parents=True)
        params_path = cache_path / "params.json"
        results_path = cache_path / "results.npz"
        with params_path.open(mode="w") as f:
            f.write(params_json)
        np.savez_compressed(results_path, **results)
        debug(f"saved {name}/{cache_key}")

    def snp_allele_counts(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask=None,
        site_class=None,
        cohort_size=None,
        random_seed=42,
    ):
        """Compute SNP allele counts. This returns the number of times each
        SNP allele was observed in the selected samples.

        Parameters
        ----------
        region : str or Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size before
            computing allele counts.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        ac : np.ndarray
            A numpy array of shape (n_variants, 4), where the first column has
            the reference allele (0) counts, the second column has the first
            alternate allele (1) counts, the third column has the second
            alternate allele (2) counts, and the fourth column has the third
            alternate allele (3) counts.

        Notes
        -----
        This computation may take some time to run, depending on your computing
        environment. Results of this computation will be cached and re-used if
        the `results_cache` parameter was set when instantiating the Ag3 class.

        """

        # change this name if you ever change the behaviour of this function,
        # to invalidate any previously cached data
        name = "ag3_snp_allele_counts_v2"

        # normalize params for consistent hash value
        if isinstance(sample_query, str):
            # resolve query to a list of integers for more cache hits
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            sample_query = np.nonzero(loc_samples)[0].tolist()
        params = dict(
            region=self.resolve_region(region).to_dict(),
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._snp_allele_counts(**params)
            self.results_cache_set(name=name, params=params, results=results)

        ac = results["ac"]
        return ac

    def _snp_allele_counts(
        self,
        *,
        region,
        sample_sets,
        sample_query,
        site_mask,
        site_class,
        cohort_size,
        random_seed,
    ):
        debug = self._log.debug

        debug("access SNP calls")
        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )
        gt = ds_snps["call_genotype"]

        debug("set up and run allele counts computation")
        gt = allel.GenotypeDaskArray(gt.data)
        ac = gt.count_alleles(max_allele=3)
        with self._dask_progress(desc="Compute SNP allele counts"):
            ac = ac.compute()

        debug("return plain numpy array")
        results = dict(ac=ac.values)

        return results

    def pca(
        self,
        region,
        n_snps,
        thin_offset=0,
        sample_sets=None,
        sample_query=None,
        site_mask="gamb_colu_arab",
        min_minor_ac=2,
        max_missing_an=0,
        n_components=20,
    ):
        """Run a principal components analysis (PCA) using biallelic SNPs from
        the selected genome region and samples.

        Parameters
        ----------
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        n_snps : int
            The desired number of SNPs to use when running the analysis.
            SNPs will be evenly thinned to approximately this number.
        thin_offset : int, optional
            Starting index for SNP thinning. Change this to repeat the analysis
            using a different set of SNPs.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        min_minor_ac : int, optional
            The minimum minor allele count. SNPs with a minor allele count
            below this value will be excluded prior to thinning.
        max_missing_an : int, optional
            The maximum number of missing allele calls to accept. SNPs with
            more than this value will be excluded prior to thinning. Set to 0
            (default) to require no missing calls.
        n_components : int, optional
            Number of components to return.

        Returns
        -------
        df_pca : pandas.DataFrame
            A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
            etc., added.
        evr : np.ndarray
            An array of explained variance ratios, one per component.

        Notes
        -----
        This computation may take some time to run, depending on your computing
        environment. Results of this computation will be cached and re-used if
        the `results_cache` parameter was set when instantiating the Ag3 class.

        """
        debug = self._log.debug

        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = "ag3_pca_v1"

        debug("normalize params for consistent hash value")
        params = dict(
            region=self.resolve_region(region).to_dict(),
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_components=n_components,
        )

        debug("try to retrieve results from the cache")
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._pca(**params)
            self.results_cache_set(name=name, params=params, results=results)

        debug("unpack results")
        coords = results["coords"]
        evr = results["evr"]

        debug("add coords to sample metadata dataframe")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
        )
        df_coords = pd.DataFrame(
            {f"PC{i + 1}": coords[:, i] for i in range(n_components)}
        )
        df_pca = pd.concat([df_samples, df_coords], axis="columns")

        return df_pca, evr

    def _pca(
        self,
        *,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_query,
        site_mask,
        min_minor_ac,
        max_missing_an,
        n_components,
    ):
        debug = self._log.debug

        debug("access SNP calls")
        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )
        debug(
            f"{ds_snps.dims['variants']:,} variants, {ds_snps.dims['samples']:,} samples"
        )

        debug("perform allele count")
        ac = self.snp_allele_counts(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )
        n_chroms = ds_snps.dims["samples"] * 2
        an_called = ac.sum(axis=1)
        an_missing = n_chroms - an_called

        debug("ascertain sites")
        ac = allel.AlleleCountsArray(ac)
        min_ref_ac = min_minor_ac
        max_ref_ac = n_chroms - min_minor_ac
        # here we choose biallelic sites involving the reference allele
        loc_sites = (
            ac.is_biallelic()
            & (ac[:, 0] >= min_ref_ac)
            & (ac[:, 0] <= max_ref_ac)
            & (an_missing <= max_missing_an)
        )
        debug(f"ascertained {np.count_nonzero(loc_sites):,} sites")

        debug("thin sites to approximately desired number")
        loc_sites = np.nonzero(loc_sites)[0]
        thin_step = max(loc_sites.shape[0] // n_snps, 1)
        loc_sites_thinned = loc_sites[thin_offset::thin_step]
        debug(f"thinned to {np.count_nonzero(loc_sites_thinned):,} sites")

        debug("access genotypes")
        gt = ds_snps["call_genotype"].data
        gt_asc = da.take(gt, loc_sites_thinned, axis=0)
        gn_asc = allel.GenotypeDaskArray(gt_asc).to_n_alt()
        with self._dask_progress(desc="Load SNP genotypes"):
            gn_asc = gn_asc.compute()

        debug("remove any sites where all genotypes are identical")
        loc_var = np.any(gn_asc != gn_asc[:, 0, np.newaxis], axis=1)
        gn_var = np.compress(loc_var, gn_asc, axis=0)
        debug(f"final shape {gn_var.shape}")

        debug("run the PCA")
        coords, model = allel.pca(gn_var, n_components=n_components)

        debug("work around sign indeterminacy")
        for i in range(n_components):
            c = coords[:, i]
            if np.abs(c.min()) > np.abs(c.max()):
                coords[:, i] = c * -1

        results = dict(coords=coords, evr=model.explained_variance_ratio_)
        return results

    def plot_pca_variance(self, evr, width=900, height=400, **kwargs):
        """Plot explained variance ratios from a principal components analysis
        (PCA) using a plotly bar plot.

        Parameters
        ----------
        evr : np.ndarray
            An array of explained variance ratios, one per component.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        **kwargs
            Passed through to px.bar().

        Returns
        -------
        fig : Figure
            A plotly figure.

        """
        debug = self._log.debug

        import plotly.express as px

        debug("prepare plotting variables")
        y = evr * 100  # convert to percent
        x = [str(i + 1) for i in range(len(y))]

        debug("set up plotting options")
        plot_kwargs = dict(
            labels={
                "x": "Principal component",
                "y": "Explained variance (%)",
            },
            template="simple_white",
            width=width,
            height=height,
        )
        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("make a bar plot")
        fig = px.bar(x=x, y=y, **plot_kwargs)

        return fig

    def plot_pca_coords(
        self,
        data,
        x="PC1",
        y="PC2",
        color=None,
        symbol=None,
        jitter_frac=0.02,
        random_seed=42,
        width=900,
        height=600,
        marker_size=10,
        **kwargs,
    ):
        """Plot sample coordinates from a principal components analysis (PCA)
        as a plotly scatter plot.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
            etc., added.
        x : str, optional
            Name of principal component to plot on the X axis.
        y : str, optional
            Name of principal component to plot on the Y axis.
        color : str, optional
            Name of column in the input dataframe to use to color the markers.
        symbol : str, optional
            Name of column in the input dataframe to use to choose marker symbols.
        jitter_frac : float, optional
            Randomly jitter points by this fraction of their range.
        random_seed : int, optional
            Random seed for jitter.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        marker_size : int, optional
            Marker size.

        Returns
        -------
        fig : Figure
            A plotly figure.

        """
        debug = self._log.debug

        import plotly.express as px

        debug(
            "set up data - copy and shuffle so that we don't get systematic over-plotting"
        )
        # TODO does the shuffling actually work?
        data = (
            data.copy().sample(frac=1, random_state=random_seed).reset_index(drop=True)
        )

        debug(
            "apply jitter if desired - helps spread out points when tightly clustered"
        )
        if jitter_frac:
            np.random.seed(random_seed)
            data[x] = jitter(data[x], jitter_frac)
            data[y] = jitter(data[y], jitter_frac)

        debug("convenience variables")
        data["country_location"] = data["country"] + " - " + data["location"]

        debug("set up plotting options")
        hover_data = [
            "partner_sample_id",
            "sample_set",
            "taxon",
            "country",
            "admin1_iso",
            "admin1_name",
            "admin2_name",
            "location",
            "year",
            "month",
        ]
        plot_kwargs = dict(
            width=width,
            height=height,
            color=color,
            symbol=symbol,
            template="simple_white",
            hover_name="sample_id",
            hover_data=hover_data,
            opacity=0.9,
            render_mode="svg",
        )

        debug("special handling for taxon color")
        if color == "taxon":
            _setup_taxon_colors(plot_kwargs)

        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("2D scatter plot")
        fig = px.scatter(data, x=x, y=y, **plot_kwargs)

        debug("tidy up")
        fig.update_layout(
            legend=dict(itemsizing="constant"),
        )
        fig.update_traces(marker={"size": marker_size})

        return fig

    def plot_pca_coords_3d(
        self,
        data,
        x="PC1",
        y="PC2",
        z="PC3",
        color=None,
        symbol=None,
        jitter_frac=0.02,
        random_seed=42,
        width=900,
        height=600,
        marker_size=5,
        **kwargs,
    ):
        """Plot sample coordinates from a principal components analysis (PCA)
        as a plotly 3D scatter plot.

        Parameters
        ----------
        data : pandas.DataFrame
            A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
            etc., added.
        x : str, optional
            Name of principal component to plot on the X axis.
        y : str, optional
            Name of principal component to plot on the Y axis.
        z : str, optional
            Name of principal component to plot on the Z axis.
        color : str, optional
            Name of column in the input dataframe to use to color the markers.
        symbol : str, optional
            Name of column in the input dataframe to use to choose marker symbols.
        jitter_frac : float, optional
            Randomly jitter points by this fraction of their range.
        random_seed : int, optional
            Random seed for jitter.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        marker_size : int, optional
            Marker size.

        Returns
        -------
        fig : Figure
            A plotly figure.

        """
        debug = self._log.debug

        import plotly.express as px

        debug(
            "set up data - copy and shuffle so that we don't get systematic over-plotting"
        )
        # TODO does this actually work?
        data = (
            data.copy().sample(frac=1, random_state=random_seed).reset_index(drop=True)
        )

        debug(
            "apply jitter if desired - helps spread out points when tightly clustered"
        )
        if jitter_frac:
            np.random.seed(random_seed)
            data[x] = jitter(data[x], jitter_frac)
            data[y] = jitter(data[y], jitter_frac)
            data[z] = jitter(data[z], jitter_frac)

        debug("convenience variables")
        data["country_location"] = data["country"] + " - " + data["location"]

        debug("set up plotting options")
        hover_data = [
            "partner_sample_id",
            "sample_set",
            "taxon",
            "country",
            "admin1_iso",
            "admin1_name",
            "admin2_name",
            "location",
            "year",
            "month",
        ]
        plot_kwargs = dict(
            width=width,
            height=height,
            hover_name="sample_id",
            hover_data=hover_data,
            color=color,
            symbol=symbol,
        )

        debug("special handling for taxon color")
        if color == "taxon":
            _setup_taxon_colors(plot_kwargs)

        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("3D scatter plot")
        fig = px.scatter_3d(data, x=x, y=y, z=z, **plot_kwargs)

        debug("tidy up")
        fig.update_layout(
            scene=dict(aspectmode="cube"),
            legend=dict(itemsizing="constant"),
        )
        fig.update_traces(marker={"size": marker_size})

        return fig

    def plot_snps_track(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask="gamb_colu_arab",
        width=800,
        height=120,
        max_snps=200_000,
        x_range=None,
        show=True,
    ):
        """Plot SNPs in a given genome region. SNPs are shown as rectangles,
        with segregating and non-segregating SNPs positioned on different levels,
        and coloured by site filter.

        Parameters
        ----------
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        site_mask : str, optional
            Site filters mask to apply.
        width : int, optional
            Width of plot in pixels (px).
        height : int, optional
            Height of plot in pixels (px).
        max_snps : int, optional
            Maximum number of SNPs to plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).
        show : bool, optional
            If True, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.


        """
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.palettes as bkpal
        import bokeh.plotting as bkplt

        debug("resolve and check region")
        region = self.resolve_region(region)
        if (
            (region.start is None)
            or (region.end is None)
            or ((region.end - region.start) > max_snps)
        ):
            raise ValueError("Region is too large, please provide a smaller region.")

        debug("compute allele counts")
        ac = allel.AlleleCountsArray(
            self.snp_allele_counts(
                region=region,
                sample_sets=sample_sets,
                sample_query=sample_query,
            )
        )
        an = ac.sum(axis=1)
        is_seg = ac.is_segregating()
        is_var = ac.is_variant()
        allelism = ac.allelism()

        debug("obtain SNP variants data")
        ds_sites = self.snp_variants(
            region=region,
        ).compute()

        debug("build a dataframe")
        pos = ds_sites["variant_position"].values
        alleles = ds_sites["variant_allele"].values.astype("U")
        cols = {
            "pos": pos,
            "allele_0": alleles[:, 0],
            "allele_1": alleles[:, 1],
            "allele_2": alleles[:, 2],
            "allele_3": alleles[:, 3],
            "ac_0": ac[:, 0],
            "ac_1": ac[:, 1],
            "ac_2": ac[:, 2],
            "ac_3": ac[:, 3],
            "an": an,
            "is_seg": is_seg,
            "is_var": is_var,
            "allelism": allelism,
            "pass_gamb_colu_arab": ds_sites[
                "variant_filter_pass_gamb_colu_arab"
            ].values,
            "pass_gamb_colu": ds_sites["variant_filter_pass_gamb_colu"].values,
            "pass_arab": ds_sites["variant_filter_pass_arab"].values,
        }
        data = pd.DataFrame(cols)

        debug("create figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        pos = data["pos"].values
        x_min = pos[0]
        x_max = pos[-1]
        if x_range is None:
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        tooltips = [
            ("Position", "$x{0,0}"),
            (
                "Alleles",
                "@allele_0 (@ac_0), @allele_1 (@ac_1), @allele_2 (@ac_2), @allele_3 (@ac_3)",
            ),
            ("No. alleles", "@allelism"),
            ("Allele calls", "@an"),
            ("Pass gamb_colu_arab", "@pass_gamb_colu_arab"),
            ("Pass gamb_colu", "@pass_gamb_colu"),
            ("Pass arab", "@pass_arab"),
        ]

        fig = bkplt.figure(
            title="SNPs",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            plot_width=width,
            plot_height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0.5, 2.5),
            tooltips=tooltips,
        )
        hover_tool = fig.select(type=bkmod.HoverTool)
        hover_tool.names = ["snps"]

        debug("plot gaps in the reference genome")
        seq = self.genome_sequence(region=region.contig).compute()
        is_n = (seq == b"N") | (seq == b"n")
        loc_n_start = ~is_n[:-1] & is_n[1:]
        loc_n_stop = is_n[:-1] & ~is_n[1:]
        n_starts = np.nonzero(loc_n_start)[0]
        n_stops = np.nonzero(loc_n_stop)[0]
        df_n_runs = pd.DataFrame(
            {"left": n_starts + 1.6, "right": n_stops + 1.4, "top": 2.5, "bottom": 0.5}
        )
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            color="#cccccc",
            source=df_n_runs,
            name="gaps",
        )

        debug("plot SNPs")
        color_pass = bkpal.Colorblind6[3]
        color_fail = bkpal.Colorblind6[5]
        data["left"] = data["pos"] - 0.4
        data["right"] = data["pos"] + 0.4
        data["bottom"] = np.where(data["is_seg"], 1.6, 0.6)
        data["top"] = data["bottom"] + 0.8
        data["color"] = np.where(data[f"pass_{site_mask}"], color_pass, color_fail)
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            color="color",
            source=data,
            name="snps",
        )
        # TODO add legend?

        debug("tidy plot")
        fig.yaxis.ticker = bkmod.FixedTicker(
            ticks=[1, 2],
        )
        fig.yaxis.major_label_overrides = {
            1: "Non-segregating",
            2: "Segregating",
        }
        fig.xaxis.axis_label = f"Contig {region.contig} position (bp)"
        fig.xaxis.ticker = bkmod.AdaptiveTicker(min_interval=1)
        fig.xaxis.minor_tick_line_color = None
        fig.xaxis[0].formatter = bkmod.NumeralTickFormatter(format="0,0")

        if show:
            bkplt.show(fig)

        return fig

    def plot_snps(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask="gamb_colu_arab",
        width=800,
        track_height=80,
        genes_height=120,
        max_snps=200_000,
        show=True,
    ):
        """Plot SNPs in a given genome region. SNPs are shown as rectangles,
        with segregating and non-segregating SNPs positioned on different levels,
        and coloured by site filter.

        Parameters
        ----------
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        site_mask : str, optional
            Site filters mask to apply.
        width : int, optional
            Width of plot in pixels (px).
        track_height : int, optional
            Height of SNPs track in pixels (px).
        genes_height : int, optional
            Height of genes track in pixels (px).
        max_snps : int, optional
            Maximum number of SNPs to show.
        show : bool, optional
            If True, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        debug("plot SNPs track")
        fig1 = self.plot_snps_track(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
            width=width,
            height=track_height,
            max_snps=max_snps,
            show=False,
        )
        fig1.xaxis.visible = False

        debug("plot genes track")
        fig2 = self.plot_genes(
            region=region,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        fig = bklay.gridplot(
            [fig1, fig2], ncols=1, toolbar_location="above", merge_tools=True
        )

        if show:
            bkplt.show(fig)

        return fig

    def aim_variants(self, aims):
        """Open ancestry informative marker variants.

        Parameters
        ----------
        aims : {'gamb_vs_colu', 'gambcolu_vs_arab'}
            Which ancestry informative markers to use.

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing AIM positions and discriminating alleles.

        """
        try:
            ds = self._cache_aim_variants[aims]
        except KeyError:
            path = f"{self._base_path}/reference/aim_defs_20220528/{aims}.zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            ds = xr.open_zarr(store, concat_characters=False)
            ds = ds.set_coords(["variant_contig", "variant_position"])
            self._cache_aim_variants[aims] = ds
        return ds.copy(deep=False)

    def _aim_calls_dataset(self, *, aims, sample_set):
        release = self._lookup_release(sample_set=sample_set)
        release_path = _release_to_path(release)
        path = f"gs://vo_agam_release/{release_path}/aim_calls_20220528/{sample_set}/{aims}.zarr"
        store = init_zarr_store(fs=self._fs, path=path)
        ds = xr.open_zarr(store=store, concat_characters=False)
        ds = ds.set_coords(["variant_contig", "variant_position", "sample_id"])
        return ds

    def aim_calls(
        self,
        aims,
        sample_sets=None,
        sample_query=None,
    ):
        """Access ancestry informative marker SNP sites, alleles and genotype
        calls.

        Parameters
        ----------
        aims : {'gamb_vs_colu', 'gambcolu_vs_arab'}
            Which ancestry informative markers to use.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing AIM SNP sites, alleles and genotype calls.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_arg(sample_sets=sample_sets)

        debug("access SNP calls and concatenate multiple sample sets and/or regions")
        ly = []
        for s in sample_sets:
            y = self._aim_calls_dataset(
                aims=aims,
                sample_set=s,
            )
            ly.append(y)

        debug("concatenate data from multiple sample sets")
        ds = xarray_concat(ly, dim=DIM_SAMPLE)

        debug("handle sample query")
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)

        return ds

    def plot_aim_heatmap(
        self,
        aims,
        sample_sets=None,
        sample_query=None,
        sort=True,
        row_height=4,
        colors="T10",
        xgap=0,
        ygap=0.5,
    ):
        """Plot a heatmap of ancestry-informative marker (AIM) genotypes.

        Parameters
        ----------
        aims : {'gamb_vs_colu', 'gambcolu_vs_arab'}
            Which ancestry informative markers to use.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sort : bool, optional
            If true (default), sort the samples by the total fraction of AIM
            alleles for the second species in the comparison.
        row_height : int, optional
            Height per sample in px.
        colors : str, optional
            Choose your favourite color palette.
        xgap : float, optional
            Creates lines between columns (variants).
        ygap : float, optional
            Creates lines between rows (samples).

        Returns
        -------
        fig : plotly.graph_objects.Figure

        """

        debug = self._log.debug

        import allel
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        debug("load AIM calls")
        ds = self.aim_calls(
            aims=aims,
            sample_sets=sample_sets,
            sample_query=sample_query,
        ).compute()
        samples = ds["sample_id"].values
        variant_contig = ds["variant_contig"].values

        debug("count variants per contig")
        contigs = ds.attrs["contigs"]
        col_widths = [
            np.count_nonzero(variant_contig == contigs.index(contig))
            for contig in contigs
        ]
        debug(col_widths)

        debug("access and transform genotypes")
        gt = allel.GenotypeArray(ds["call_genotype"].values)
        gn = gt.to_n_alt(fill=-1)

        if sort:
            debug("sort by AIM fraction")
            ac = np.sum(gt == 1, axis=(0, 2))
            an = np.sum(gt >= 0, axis=(0, 2))
            af = ac / an
            ix_sorted = np.argsort(af)
            gn = np.take(gn, ix_sorted, axis=1)
            samples = np.take(samples, ix_sorted, axis=0)

        debug("set up colors")
        # https://en.wiktionary.org/wiki/abandon_hope_all_ye_who_enter_here
        if colors.lower() == "plotly":
            palette = px.colors.qualitative.Plotly
            color_gc = palette[3]
            color_gc_a = palette[9]
            color_a = palette[2]
            color_g = palette[0]
            color_g_c = palette[9]
            color_c = palette[1]
            color_m = "white"
        elif colors.lower() == "set1":
            palette = px.colors.qualitative.Set1
            color_gc = palette[3]
            color_gc_a = palette[4]
            color_a = palette[2]
            color_g = palette[1]
            color_g_c = palette[5]
            color_c = palette[0]
            color_m = "white"
        elif colors.lower() == "g10":
            palette = px.colors.qualitative.G10
            color_gc = palette[4]
            color_gc_a = palette[2]
            color_a = palette[3]
            color_g = palette[0]
            color_g_c = palette[2]
            color_c = palette[8]
            color_m = "white"
        elif colors.lower() == "t10":
            palette = px.colors.qualitative.T10
            color_gc = palette[6]
            color_gc_a = palette[5]
            color_a = palette[4]
            color_g = palette[0]
            color_g_c = palette[5]
            color_c = palette[2]
            color_m = "white"
        else:
            raise ValueError("unsupported colors")
        if aims == "gambcolu_vs_arab":
            colors = [color_m, color_gc, color_gc_a, color_a]
        else:
            colors = [color_m, color_g, color_g_c, color_c]
        species = aims.split("_vs_")

        debug("create subplots")
        fig = make_subplots(
            rows=1,
            cols=len(contigs),
            shared_yaxes=True,
            column_titles=contigs,
            row_titles=None,
            column_widths=col_widths,
            x_title="Variants",
            y_title="Samples",
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
        )

        for j, contig in enumerate(contigs):
            debug(f"plot {contig}")
            loc_contig = variant_contig == j
            gn_contig = np.compress(loc_contig, gn, axis=0)
            fig.add_trace(
                go.Heatmap(
                    y=samples,
                    z=gn_contig.T,
                    # construct a discrete color scale
                    # https://plotly.com/python/colorscales/#constructing-a-discrete-or-discontinuous-color-scale
                    colorscale=[
                        [0 / 4, colors[0]],
                        [1 / 4, colors[0]],
                        [1 / 4, colors[1]],
                        [2 / 4, colors[1]],
                        [2 / 4, colors[2]],
                        [3 / 4, colors[2]],
                        [3 / 4, colors[3]],
                        [4 / 4, colors[3]],
                    ],
                    zmin=-1.5,
                    zmax=2.5,
                    xgap=xgap,
                    ygap=ygap,  # this creates faint lines between rows
                    colorbar=dict(
                        title="AIM genotype",
                        tickmode="array",
                        tickvals=[-1, 0, 1, 2],
                        ticktext=[
                            "missing",
                            f"{species[0]}/{species[0]}",
                            f"{species[0]}/{species[1]}",
                            f"{species[1]}/{species[1]}",
                        ],
                        len=100,
                        lenmode="pixels",
                        y=1,
                        yanchor="top",
                        outlinewidth=1,
                        outlinecolor="black",
                    ),
                    hovertemplate=dedent(
                        """
                        Variant index: %{x}<br>
                        Sample: %{y}<br>
                        Genotype: %{z}
                        <extra></extra>
                    """
                    ),
                ),
                row=1,
                col=j + 1,
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=[],
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=[],
        )

        fig.update_layout(
            title=f"AIMs - {aims}",
            height=max(300, row_height * len(samples) + 100),
        )

        return fig

    def _block_jackknife_cohort_diversity_stats(
        self, *, cohort_label, ac, n_jack, confidence_level
    ):
        debug = self._log.debug

        debug("set up for diversity calculations")
        n_sites = ac.shape[0]
        ac = allel.AlleleCountsArray(ac)
        n = ac.sum(axis=1).max()  # number of chromosomes sampled
        n_sites = min(n_sites, ac.shape[0])  # number of sites
        block_length = n_sites // n_jack  # number of sites in each block
        n_sites_j = n_sites - block_length  # number of sites in each jackknife resample

        debug("compute scaling constants")
        a1 = np.sum(1 / np.arange(1, n))
        a2 = np.sum(1 / (np.arange(1, n) ** 2))
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - (1 / a1)
        c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1**2))
        e1 = c1 / a1
        e2 = c2 / (a1**2 + a2)

        debug(
            "compute some intermediates ahead of time, to minimise computation during jackknife resampling"
        )
        mpd_data = allel.mean_pairwise_difference(ac, fill=0)
        # N.B., here we compute the number of segregating sites as the number
        # of alleles minus 1. This follows the sgkit and tskit implementations,
        # and is different from scikit-allel.
        seg_data = ac.allelism() - 1

        debug("compute estimates from all data")
        theta_pi_abs_data = np.sum(mpd_data)
        theta_pi_data = theta_pi_abs_data / n_sites
        S_data = np.sum(seg_data)
        theta_w_abs_data = S_data / a1
        theta_w_data = theta_w_abs_data / n_sites
        d_data = theta_pi_abs_data - theta_w_abs_data
        d_stdev_data = np.sqrt((e1 * S_data) + (e2 * S_data * (S_data - 1)))
        tajima_d_data = d_data / d_stdev_data

        debug("set up for jackknife resampling")
        jack_theta_pi = []
        jack_theta_w = []
        jack_tajima_d = []

        debug("begin jackknife resampling")
        for i in range(n_jack):

            # locate block to delete
            block_start = i * block_length
            block_stop = block_start + block_length
            loc_j = np.ones(n_sites, dtype=bool)
            loc_j[block_start:block_stop] = False
            assert np.count_nonzero(loc_j) == n_sites_j

            # resample data and compute statistics

            # theta_pi
            mpd_j = mpd_data[loc_j]
            theta_pi_abs_j = np.sum(mpd_j)
            theta_pi_j = theta_pi_abs_j / n_sites_j
            jack_theta_pi.append(theta_pi_j)

            # theta_w
            seg_j = seg_data[loc_j]
            S_j = np.sum(seg_j)
            theta_w_abs_j = S_j / a1
            theta_w_j = theta_w_abs_j / n_sites_j
            jack_theta_w.append(theta_w_j)

            # tajima_d
            d_j = theta_pi_abs_j - theta_w_abs_j
            d_stdev_j = np.sqrt((e1 * S_j) + (e2 * S_j * (S_j - 1)))
            tajima_d_j = d_j / d_stdev_j
            jack_tajima_d.append(tajima_d_j)

        # calculate jackknife stats
        (
            theta_pi_estimate,
            theta_pi_bias,
            theta_pi_std_err,
            theta_pi_ci_err,
            theta_pi_ci_low,
            theta_pi_ci_upp,
        ) = jackknife_ci(
            stat_data=theta_pi_data,
            jack_stat=jack_theta_pi,
            confidence_level=confidence_level,
        )
        (
            theta_w_estimate,
            theta_w_bias,
            theta_w_std_err,
            theta_w_ci_err,
            theta_w_ci_low,
            theta_w_ci_upp,
        ) = jackknife_ci(
            stat_data=theta_w_data,
            jack_stat=jack_theta_w,
            confidence_level=confidence_level,
        )
        (
            tajima_d_estimate,
            tajima_d_bias,
            tajima_d_std_err,
            tajima_d_ci_err,
            tajima_d_ci_low,
            tajima_d_ci_upp,
        ) = jackknife_ci(
            stat_data=tajima_d_data,
            jack_stat=jack_tajima_d,
            confidence_level=confidence_level,
        )

        return dict(
            cohort=cohort_label,
            theta_pi=theta_pi_data,
            theta_pi_estimate=theta_pi_estimate,
            theta_pi_bias=theta_pi_bias,
            theta_pi_std_err=theta_pi_std_err,
            theta_pi_ci_err=theta_pi_ci_err,
            theta_pi_ci_low=theta_pi_ci_low,
            theta_pi_ci_upp=theta_pi_ci_upp,
            theta_w=theta_w_data,
            theta_w_estimate=theta_w_estimate,
            theta_w_bias=theta_w_bias,
            theta_w_std_err=theta_w_std_err,
            theta_w_ci_err=theta_w_ci_err,
            theta_w_ci_low=theta_w_ci_low,
            theta_w_ci_upp=theta_w_ci_upp,
            tajima_d=tajima_d_data,
            tajima_d_estimate=tajima_d_estimate,
            tajima_d_bias=tajima_d_bias,
            tajima_d_std_err=tajima_d_std_err,
            tajima_d_ci_err=tajima_d_ci_err,
            tajima_d_ci_low=tajima_d_ci_low,
            tajima_d_ci_upp=tajima_d_ci_upp,
        )

    def cohort_diversity_stats(
        self,
        cohort,
        cohort_size,
        region,
        site_mask,
        site_class,
        sample_sets=None,
        random_seed=42,
        n_jack=200,
        confidence_level=0.95,
    ):
        """Compute genetic diversity summary statistics for a cohort of
        individuals.

        Parameters
        ----------
        cohort : str or (str, str)
            Either a string giving one of the predefined cohort labels, or a
            pair of strings giving a custom cohort label and a sample query to
            select samples in the cohort.
        cohort_size : int
            Number of individuals to use for computation of summary statistics.
            If the cohort is larger than this size, it will be randomly
            down-sampled.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        random_seed : int, optional
            Seed for random number generator.
        n_jack : int, optional
            Number of blocks to divide the data into for the block jackknife
            estimation of confidence intervals. N.B., larger is not necessarily
            better.
        confidence_level : float, optional
            Confidence level to use for confidence interval calculation. 0.95
            means 95% confidence interval.

        Returns
        -------
        stats : pandas.Series
            A series with summary statistics and their confidence intervals.

        """

        debug = self._log.debug

        debug("process cohort parameter")
        cohort_query = None
        if isinstance(cohort, str):
            # assume it is one of the predefined cohorts
            cohort_label = cohort
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            cohort_cols = [c for c in df_samples.columns if c.startswith("cohort_")]
            for c in cohort_cols:
                if cohort in set(df_samples[c]):
                    cohort_query = f"{c} == '{cohort}'"
                    break
            if cohort_query is None:
                raise ValueError(f"unknown cohort: {cohort}")

        elif isinstance(cohort, (list, tuple)) and len(cohort) == 2:
            cohort_label, cohort_query = cohort

        else:
            raise TypeError(r"invalid cohort parameter: {cohort!r}")

        debug("access allele counts")
        ac = self.snp_allele_counts(
            region=region,
            site_mask=site_mask,
            site_class=site_class,
            sample_query=cohort_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        debug("compute diversity stats")
        stats = self._block_jackknife_cohort_diversity_stats(
            cohort_label=cohort_label,
            ac=ac,
            n_jack=n_jack,
            confidence_level=confidence_level,
        )

        debug("compute some extra cohort variables")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=cohort_query
        )
        extra_fields = [
            ("taxon", "unique"),
            ("year", "unique"),
            ("month", "unique"),
            ("country", "unique"),
            ("admin1_iso", "unique"),
            ("admin1_name", "unique"),
            ("admin2_name", "unique"),
            ("longitude", "mean"),
            ("latitude", "mean"),
        ]
        for field, agg in extra_fields:
            if agg == "unique":
                vals = df_samples[field].sort_values().unique()
                if len(vals) == 0:
                    val = np.nan
                elif len(vals) == 1:
                    val = vals[0]
                else:
                    val = vals.tolist()
            elif agg == "mean":
                vals = df_samples[field]
                if len(vals) == 0:
                    val = np.nan
                else:
                    val = np.mean(vals)
            else:
                val = np.nan
            stats[field] = val

        return pd.Series(stats)

    def diversity_stats(
        self,
        cohorts,
        cohort_size,
        region,
        site_mask,
        site_class,
        sample_query=None,
        sample_sets=None,
        random_seed=42,
        n_jack=200,
        confidence_level=0.95,
    ):
        """Compute genetic diversity summary statistics for multiple cohorts.

        Parameters
        ----------
        cohorts : str or dict
            Either a string giving one of the predefined cohort columns, or a
            dictionary mapping cohort labels to sample queries.
        cohort_size : int
            Number of individuals to use for computation of summary statistics.
            If the cohort is larger than this size, it will be randomly
            down-sampled.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        site_class : str, optional
            Select sites belonging to one of the following classes: CDS_DEG_4,
            (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
            degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
            INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
            longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
            5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
            splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
            untranslated region), INTERGENIC (intergenic, more than 10 kbp from
            a gene).
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        random_seed : int, optional
            Seed for random number generator.
        n_jack : int, optional
            Number of blocks to divide the data into for the block jackknife
            estimation of confidence intervals. N.B., larger is not necessarily
            better.
        confidence_level : float, optional
            Confidence level to use for confidence interval calculation. 0.95
            means 95% confidence interval.

        Returns
        -------
        df_stats : pandas.DataFrame
            A DataFrame where each row provides summary statistics and their
            confidence intervals for a single cohort.

        """
        debug = self._log.debug
        info = self._log.info

        debug("set up cohorts")
        if isinstance(cohorts, dict):
            # user has supplied a custom dictionary mapping cohort identifiers
            # to pandas queries
            cohort_queries = cohorts

        elif isinstance(cohorts, str):
            # user has supplied one of the predefined cohort sets

            df_samples = self.sample_metadata(
                sample_sets=sample_sets, sample_query=sample_query
            )

            # determine column in dataframe - allow abbreviation
            if cohorts.startswith("cohort_"):
                cohorts_col = cohorts
            else:
                cohorts_col = "cohort_" + cohorts
            if cohorts_col not in df_samples.columns:
                raise ValueError(f"{cohorts_col!r} is not a known cohort set")

            # find cohort labels and build queries dictionary
            cohort_labels = sorted(df_samples[cohorts_col].dropna().unique())
            cohort_queries = {coh: f"{cohorts_col} == '{coh}'" for coh in cohort_labels}

        else:
            raise TypeError("cohorts parameter should be dict or str")

        debug("handle sample_query parameter")
        if sample_query is not None:
            cohort_queries = {
                cohort_label: f"({cohort_query}) and ({sample_query})"
                for cohort_label, cohort_query in cohort_queries.items()
            }

        debug("check cohort sizes, drop any cohorts which are too small")
        cohort_queries_checked = dict()
        for cohort_label, cohort_query in cohort_queries.items():
            df_cohort_samples = self.sample_metadata(
                sample_sets=sample_sets, sample_query=cohort_query
            )
            n_samples = len(df_cohort_samples)
            if n_samples < cohort_size:
                info(
                    f"cohort ({cohort_label}) has insufficient samples ({n_samples}) for requested cohort size ({cohort_size}), dropping"  # noqa
                )  # noqa
            else:
                cohort_queries_checked[cohort_label] = cohort_query

        debug("compute diversity stats for cohorts")
        all_stats = []
        for cohort_label, cohort_query in cohort_queries_checked.items():
            stats = self.cohort_diversity_stats(
                cohort=(cohort_label, cohort_query),
                cohort_size=cohort_size,
                region=region,
                site_mask=site_mask,
                site_class=site_class,
                sample_sets=sample_sets,
                random_seed=random_seed,
                n_jack=n_jack,
                confidence_level=confidence_level,
            )
            all_stats.append(stats)
        df_stats = pd.DataFrame(all_stats)

        return df_stats

    def plot_diversity_stats(
        self,
        df_stats,
        color=None,
        bar_plot_height=450,
        bar_width=30,
        scatter_plot_height=500,
        scatter_plot_width=500,
        template="plotly_white",
        plot_kwargs=None,
    ):
        """Plot diversity statistics.

        Parameters
        ----------
        df_stats : pandas.DataFrame
            Output from diversity_stats().
        color : str, optional
            Column to color by.
        bar_plot_height : int, optional
            Height of bar plots in pixels (px).
        bar_width : int, optional
            Width per bar in pixels (px).
        scatter_plot_height : int, optional
            Height of scatter plot in pixels (px).
        scatter_plot_width : int, optional
            Width of scatter plot in pixels (px).
        template : str, optional
            Plotly template.
        plot_kwargs : dict, optional
            Extra plotting parameters

        """
        debug = self._log.debug
        import plotly.express as px

        debug("set up common plotting parameters")
        if plot_kwargs is None:
            plot_kwargs = dict()
        default_plot_kwargs = dict(
            hover_name="cohort",
            hover_data=[
                "taxon",
                "country",
                "admin1_iso",
                "admin1_name",
                "admin2_name",
                "longitude",
                "latitude",
                "year",
                "month",
            ],
            labels={
                "theta_pi_estimate": r"$\widehat{\theta}_{\pi}$",
                "theta_w_estimate": r"$\widehat{\theta}_{w}$",
                "tajima_d_estimate": r"$D$",
                "cohort": "Cohort",
                "taxon": "Taxon",
                "country": "Country",
            },
        )
        if color == "taxon":
            _setup_taxon_colors(plot_kwargs=default_plot_kwargs)
        default_plot_kwargs.update(plot_kwargs)
        plot_kwargs = default_plot_kwargs
        bar_plot_width = 300 + bar_width * len(df_stats)

        debug("nucleotide diversity bar plot")
        fig = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_pi_estimate",
            error_y="theta_pi_ci_err",
            title="Nucleotide diversity",
            color=color,
            height=bar_plot_height,
            width=bar_plot_width,
            template=template,
            **plot_kwargs,
        )
        fig.show()

        debug("Watterson's estimator bar plot")
        fig = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_w_estimate",
            error_y="theta_w_ci_err",
            title="Watterson's estimator",
            color=color,
            height=bar_plot_height,
            width=bar_plot_width,
            template=template,
            **plot_kwargs,
        )
        fig.show()

        debug("Tajima's D bar plot")
        fig = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="tajima_d_estimate",
            error_y="tajima_d_ci_err",
            title="Tajima's D",
            color=color,
            height=bar_plot_height,
            width=bar_plot_width,
            template=template,
            **plot_kwargs,
        )
        fig.show()

        debug("scatter plot comparing diversity estimators")
        fig = px.scatter(
            data_frame=df_stats,
            x="theta_pi_estimate",
            y="theta_w_estimate",
            error_x="theta_pi_ci_err",
            error_y="theta_w_ci_err",
            title="Diversity estimators",
            color=color,
            width=scatter_plot_width,
            height=scatter_plot_height,
            template=template,
            **plot_kwargs,
        )
        fig.show()

    def plot_samples_interactive_map(
        self,
        sample_sets=None,
        sample_query=None,
        basemap=None,
        center=(-2, 20),
        zoom=3,
        min_samples=1,
    ):
        """Plot an interactive map showing sampling locations using ipyleaflet.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        basemap : dict
            Basemap description coming from ipyleaflet.basemaps.
        center : tuple of int, optional
            Location to center the map.
        zoom : int, optional
            Initial zoom level.
        min_samples : int, optional
            Minimum number of samples required to show a marker for a given
            location.

        Returns
        -------
        samples_map : ipyleaflet.Map
            Ipyleaflet map widget.

        """
        debug = self._log.debug

        import ipyleaflet

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("compute locations")
        pivot_location_taxon = df_samples.pivot_table(
            index=[
                "country",
                "admin1_iso",
                "admin1_name",
                "admin2_name",
                "location",
                "latitude",
                "longitude",
            ],
            columns=["taxon"],
            values="sample_id",
            aggfunc="count",
            fill_value=0,
        )

        debug("create a map")
        if basemap is None:
            basemap = ipyleaflet.basemaps.Esri.WorldImagery
        samples_map = ipyleaflet.Map(
            center=center,
            zoom=zoom,
            basemap=basemap,
        )
        samples_map.add_control(ipyleaflet.ScaleControl(position="bottomleft"))
        # make the map a bit taller than the default
        samples_map.layout.height = "500px"

        debug("add markers")
        taxa = df_samples["taxon"].dropna().sort_values().unique()
        for _, row in pivot_location_taxon.reset_index().iterrows():
            title = (
                f"Location: {row.location} ({row.latitude:.3f}, {row.longitude:.3f})"
            )
            title += f"\nAdmin level 2: {row.admin2_name}"
            title += f"\nAdmin level 1: {row.admin1_name} ({row.admin1_iso})"
            title += f"\nCountry: {row.country}"
            title += "\nNo. specimens: "
            all_n = 0
            for taxon in taxa:
                n = row[taxon]
                all_n += n
                if n > 0:
                    title += f"{n} {taxon}; "
            if all_n >= min_samples:
                marker = ipyleaflet.Marker(
                    location=(row.latitude, row.longitude),
                    draggable=False,
                    title=title,
                )
                samples_map.add_layer(marker)

        return samples_map

    def count_samples(
        self,
        sample_sets=None,
        sample_query=None,
        index=(
            "country",
            "admin1_iso",
            "admin1_name",
            "admin2_name",
            "year",
        ),
        columns="taxon",
    ):
        """Create a pivot table showing numbers of samples available by space,
        time and taxon.

        Parameters
        ----------
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        index : str or tuple of str
            Sample metadata columns to use for the index.
        columns : str or tuple of str
            Sample metadata columns to use for the columns.

        Returns
        -------
        df : pandas.DataFrame
            Pivot table of sample counts.

        """
        debug = self._log.debug

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        debug("create pivot table")
        df_pivot = df_samples.pivot_table(
            index=index,
            columns=columns,
            values="sample_id",
            aggfunc="count",
            fill_value=0,
        )

        return df_pivot

    def _lookup_sample(self, sample, sample_set=None):
        df_samples = self.sample_metadata(sample_sets=sample_set).set_index("sample_id")
        sample_rec = None
        if isinstance(sample, str):
            sample_rec = df_samples.loc[sample]
        elif isinstance(sample, int):
            sample_rec = df_samples.iloc[sample]
        else:
            type_error(name="sample", value=sample, expectation=(str, int))
        return sample_rec

    def _plot_heterozygosity_track(
        self,
        *,
        sample_id,
        sample_set,
        windows,
        counts,
        region,
        window_size,
        y_max,
        width,
        height,
        circle_kwargs,
        show,
        x_range,
    ):
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        region = self.resolve_region(region)

        # pos axis
        window_pos = windows.mean(axis=1)

        # het axis
        window_het = counts / window_size

        # determine plotting limits
        if x_range is None:
            if region.start is not None:
                x_min = region.start
            else:
                x_min = 0
            if region.end is not None:
                x_max = region.end
            else:
                x_max = len(self.genome_sequence(region.contig))
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        debug("create a figure for plotting")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=f"{sample_id} ({sample_set})",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            plot_width=width,
            plot_height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, y_max),
        )

        debug("plot heterozygosity")
        data = pd.DataFrame(
            {
                "position": window_pos,
                "heterozygosity": window_het,
            }
        )
        if circle_kwargs is None:
            circle_kwargs = dict()
        circle_kwargs.setdefault("size", 3)
        fig.circle(x="position", y="heterozygosity", source=data, **circle_kwargs)

        debug("tidy up the plot")
        fig.yaxis.axis_label = "Heterozygosity (bp⁻¹)"
        _bokeh_style_genome_xaxis(fig, region.contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_heterozygosity_track(
        self,
        sample,
        region,
        site_mask,
        window_size,
        sample_set=None,
        y_max=0.03,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=200,
        circle_kwargs=None,
        show=True,
        x_range=None,
    ):
        """Plot windowed heterozygosity for a single sample over a genome
        region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        window_size : int
            Number of sites per window.
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        y_max : float, optional
            Y axis limit.
        width : int, optional
            Plot width in pixels (px).
        height : int, optional
            Plot height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        show : bool, optional
            If true, show the plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """
        debug = self._log.debug

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
        )

        debug("plot heterozygosity")
        fig = self._plot_heterozygosity_track(
            sample_id=sample_id,
            sample_set=sample_set,
            windows=windows,
            counts=counts,
            region=region,
            window_size=window_size,
            y_max=y_max,
            width=width,
            height=height,
            circle_kwargs=circle_kwargs,
            show=show,
            x_range=x_range,
        )

        return fig

    def plot_heterozygosity(
        self,
        sample,
        region,
        site_mask,
        window_size,
        sample_set=None,
        y_max=0.03,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=170,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        circle_kwargs=None,
        show=True,
    ):
        """Plot windowed heterozygosity for a single sample over a genome
        region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        window_size : int
            Number of sites per window.
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        y_max : float, optional
            Y axis limit.
        width : int, optional
            Plot width in pixels (px).
        track_height : int, optional
            Heterozygosity track height in pixels (px).
        genes_height : int, optional
            Genes track height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """

        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        # normalise to support multiple samples
        if isinstance(sample, (list, tuple)):
            samples = sample
        else:
            samples = [sample]

        debug("plot first sample track")
        fig1 = self.plot_heterozygosity_track(
            sample=samples[0],
            sample_set=sample_set,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            y_max=y_max,
            width=width,
            height=track_height,
            circle_kwargs=circle_kwargs,
            show=False,
        )
        fig1.xaxis.visible = False
        figs = [fig1]

        debug("plot remaining sample tracks")
        for sample in samples[1:]:
            fig_het = self.plot_heterozygosity_track(
                sample=sample,
                sample_set=sample_set,
                region=region,
                site_mask=site_mask,
                window_size=window_size,
                y_max=y_max,
                width=width,
                height=track_height,
                circle_kwargs=circle_kwargs,
                show=False,
                x_range=fig1.x_range,
            )
            fig_het.xaxis.visible = False
            figs.append(fig_het)

        debug("plot genes track")
        fig_genes = self.plot_genes(
            region=region,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )
        figs.append(fig_genes)

        debug("combine plots into a single figure")
        fig_all = bklay.gridplot(
            figs, ncols=1, toolbar_location="above", merge_tools=True
        )

        if show:
            bkplt.show(fig_all)

        return fig_all

    def _sample_count_het(
        self,
        sample,
        region,
        site_mask,
        window_size,
        sample_set=None,
    ):
        debug = self._log.debug

        region = self.resolve_region(region)

        debug("access sample metadata, look up sample")
        sample_rec = self._lookup_sample(sample=sample, sample_set=sample_set)
        sample_id = sample_rec.name  # sample_id
        sample_set = sample_rec["sample_set"]

        debug("access SNPs, select data for sample")
        ds_snps = self.snp_calls(
            region=region, sample_sets=sample_set, site_mask=site_mask
        )
        ds_snps_sample = ds_snps.set_index(samples="sample_id").sel(samples=sample_id)

        # snp positions
        pos = ds_snps_sample["variant_position"].values

        # access genotypes
        gt = allel.GenotypeDaskVector(ds_snps_sample["call_genotype"].data)

        # compute het
        with self._dask_progress(desc="Compute heterozygous genotypes"):
            is_het = gt.is_het().compute()

        # compute window coordinates
        windows = allel.moving_statistic(
            values=pos,
            statistic=lambda x: [x[0], x[-1]],
            size=window_size,
        )

        # compute windowed heterozygosity
        counts = allel.moving_statistic(
            values=is_het,
            statistic=np.sum,
            size=window_size,
        )

        return sample_id, sample_set, windows, counts

    def roh_hmm(
        self,
        sample,
        region,
        site_mask,
        window_size,
        sample_set=None,
        phet_roh=0.001,
        phet_nonroh=(0.003, 0.01),
        transition=1e-3,
    ):
        """Infer runs of homozygosity for a single sample over a genome region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        window_size : int
            Number of sites per window.
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        phet_roh: float, optional
            Probability of observing a heterozygote in a ROH.
        phet_nonroh: tuple of floats, optional
            One or more probabilities of observing a heterozygote outside a ROH.
        transition: float, optional
           Probability of moving between states. A larger window size may call
           for a larger transitional probability.

        Returns
        -------
        df_roh : pandas.DataFrame
            A DataFrame where each row provides data about a single run of
            homozygosity.

        """
        debug = self._log.debug

        region = self.resolve_region(region)

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
        )

        debug("compute runs of homozygosity")
        df_roh = _roh_hmm_predict(
            windows=windows,
            counts=counts,
            phet_roh=phet_roh,
            phet_nonroh=phet_nonroh,
            transition=transition,
            window_size=window_size,
            sample_id=sample_id,
            contig=region.contig,
        )

        return df_roh

    def plot_roh_track(
        self,
        df_roh,
        region,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=100,
        show=True,
        x_range=None,
        title="Runs of homozygosity",
    ):
        debug = self._log.debug

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        debug("handle region parameter - this determines the genome region to plot")
        region = self.resolve_region(region)
        contig = region.contig
        start = region.start
        end = region.end
        if start is None:
            start = 0
        if end is None:
            end = len(self.genome_sequence(contig))

        debug("define x axis range")
        if x_range is None:
            x_range = bkmod.Range1d(start, end, bounds="auto")

        debug(
            "we're going to plot each gene as a rectangle, so add some additional columns"
        )
        data = df_roh.copy()
        data["bottom"] = 0.2
        data["top"] = 0.8

        debug("make a figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=title,
            plot_width=width,
            plot_height=height,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "tap",
                "hover",
            ],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            x_range=x_range,
        )

        debug("now plot the ROH as rectangles")
        fig.quad(
            bottom="bottom",
            top="top",
            left="roh_start",
            right="roh_stop",
            source=data,
            line_width=0.5,
            fill_alpha=0.5,
        )

        debug("tidy up the plot")
        fig.y_range = bkmod.Range1d(0, 1)
        fig.ygrid.visible = False
        fig.yaxis.ticker = []
        _bokeh_style_genome_xaxis(fig, region.contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_roh(
        self,
        sample,
        region,
        site_mask,
        window_size,
        sample_set=None,
        phet_roh=0.001,
        phet_nonroh=(0.003, 0.01),
        transition=1e-3,
        y_max=0.03,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        heterozygosity_height=170,
        roh_height=50,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
        circle_kwargs=None,
        show=True,
    ):
        """Plot windowed heterozygosity and inferred runs of homozygosity for a
        single sample over a genome region.

        Parameters
        ----------
        sample : str or int
            Sample identifier or index within sample set.
        region : str
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280") or
            genomic region defined with coordinates (e.g.,
            "2L:44989425-44998059").
        site_mask : {"gamb_colu_arab", "gamb_colu", "arab"}
            Site filters mask to apply.
        window_size : int
            Number of sites per window.
        sample_set : str, optional
            Sample set identifier. Not needed if sample parameter gives a sample
            identifier.
        phet_roh: float, optional
            Probability of observing a heterozygote in a ROH.
        phet_nonroh: tuple of floats, optional
            One or more probabilities of observing a heterozygote outside a ROH.
        transition: float, optional
           Probability of moving between states. A larger window size may call
           for a larger transitional probability.
        y_max : float, optional
            Y axis limit.
        width : int, optional
            Plot width in pixels (px).
        heterozygosity_height : int, optional
            Heterozygosity track height in pixels (px).
        roh_height : int, optional
            ROH track height in pixels (px).
        genes_height : int, optional
            Genes track height in pixels (px).
        circle_kwargs : dict, optional
            Passed through to bokeh circle() function.
        show : bool, optional
            If true, show the plot.

        Returns
        -------
        fig : Figure
            Bokeh figure.

        """

        debug = self._log.debug

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        region = self.resolve_region(region)

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
        )

        debug("plot_heterozygosity track")
        fig_het = self._plot_heterozygosity_track(
            sample_id=sample_id,
            sample_set=sample_set,
            windows=windows,
            counts=counts,
            region=region,
            window_size=window_size,
            y_max=y_max,
            width=width,
            height=heterozygosity_height,
            circle_kwargs=circle_kwargs,
            show=False,
            x_range=None,
        )
        fig_het.xaxis.visible = False
        figs = [fig_het]

        debug("compute runs of homozygosity")
        df_roh = _roh_hmm_predict(
            windows=windows,
            counts=counts,
            phet_roh=phet_roh,
            phet_nonroh=phet_nonroh,
            transition=transition,
            window_size=window_size,
            sample_id=sample_id,
            contig=region.contig,
        )

        debug("plot roh track")
        fig_roh = self.plot_roh_track(
            df_roh,
            region=region,
            width=width,
            height=roh_height,
            show=False,
            x_range=fig_het.x_range,
        )
        fig_roh.xaxis.visible = False
        figs.append(fig_roh)

        debug("plot genes track")
        fig_genes = self.plot_genes(
            region=region,
            width=width,
            height=genes_height,
            x_range=fig_het.x_range,
            show=False,
        )
        figs.append(fig_genes)

        debug("combine plots into a single figure")
        fig_all = bklay.gridplot(
            figs, ncols=1, toolbar_location="above", merge_tools=True
        )

        if show:
            bkplt.show(fig_all)

        return fig_all

    def h12_calibration(
        self,
        contig,
        analysis,
        sample_query=None,
        sample_sets=None,
        cohort_size=30,
        window_sizes=(100, 200, 500, 1000, 2000, 5000, 10000, 20000),
        random_seed=42,
    ):
        """Generate h12 GWSS calibration data for different window sizes.

        Parameters
        ----------
        contig: str
            Chromosome arm (e.g., "2L")
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        window_sizes : int or list of int, optional
            The sizes of windows used to calculate h12 over. Multiple window
            sizes should be used to calibrate the optimal size for h12 analysis.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        calibration runs : list of numpy.ndarrays
            A list of h12 calibration run arrays for each window size, containing
            values and percentiles.

        """

        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = "ag3_h12_calibration_v1"

        params = dict(
            contig=contig,
            analysis=analysis,
            window_sizes=window_sizes,
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            calibration_runs = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            calibration_runs = self._h12_calibration(**params)
            self.results_cache_set(name=name, params=params, results=calibration_runs)

        return calibration_runs

    def _h12_calibration(
        self,
        contig,
        analysis,
        sample_query,
        sample_sets,
        cohort_size,
        window_sizes,
        random_seed,
    ):
        # access haplotypes
        ds_haps = self.haplotypes(
            region=contig,
            sample_sets=sample_sets,
            sample_query=sample_query,
            analysis=analysis,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        calibration_runs = dict()
        for window_size in self._progress(window_sizes, desc="Compute H12"):
            h1, h12, h123, h2_h1 = allel.moving_garud_h(ht, size=window_size)
            calibration_runs[str(window_size)] = h12

        return calibration_runs

    def plot_h12_calibration(
        self,
        contig,
        analysis,
        sample_query=None,
        sample_sets=None,
        cohort_size=30,
        window_sizes=(100, 200, 500, 1000, 2000, 5000, 10000, 20000),
        random_seed=42,
        title=None,
    ):
        """Plot h12 GWSS calibration data for different window sizes.

        Parameters
        ----------
        contig: str
            Chromosome arm (e.g., "2L")
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        window_sizes : int or list of int, optional
            The sizes of windows used to calculate h12 over. Multiple window
            sizes should be used to calibrate the optimal size for h12 analysis.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.

        Returns
        -------
        fig : figure
            A plot showing h12 calibration run percentiles for different window
            sizes.

        """

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        # get H12 values
        calibration_runs = self.h12_calibration(
            contig=contig,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            window_sizes=window_sizes,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        # compute summaries
        q50 = [np.median(calibration_runs[str(window)]) for window in window_sizes]
        q25 = [
            np.percentile(calibration_runs[str(window)], 25) for window in window_sizes
        ]
        q75 = [
            np.percentile(calibration_runs[str(window)], 75) for window in window_sizes
        ]
        q05 = [
            np.percentile(calibration_runs[str(window)], 5) for window in window_sizes
        ]
        q95 = [
            np.percentile(calibration_runs[str(window)], 95) for window in window_sizes
        ]

        # make plot
        fig = bkplt.figure(plot_width=700, plot_height=400, x_axis_type="log")
        fig.patch(
            window_sizes + window_sizes[::-1],
            q75 + q25[::-1],
            alpha=0.75,
            line_width=2,
            legend_label="25-75%",
        )
        fig.patch(
            window_sizes + window_sizes[::-1],
            q95 + q05[::-1],
            alpha=0.5,
            line_width=2,
            legend_label="5-95%",
        )
        fig.line(
            window_sizes, q50, line_color="black", line_width=4, legend_label="median"
        )
        fig.circle(window_sizes, q50, color="black", fill_color="black", size=8)

        fig.xaxis.ticker = window_sizes
        fig.x_range = bkmod.Range1d(window_sizes[0], window_sizes[-1])
        if title is None:
            title = sample_query
        fig.title = title
        bkplt.show(fig)

    def h12_gwss(
        self,
        contig,
        analysis,
        window_size,
        sample_sets=None,
        sample_query=None,
        cohort_size=30,
        random_seed=42,
    ):
        """Run h12 GWSS.

        Parameters
        ----------
        contig: str
            Chromosome arm (e.g., "2L")
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.
        window_size : int
            The size of windows used to calculate h12 over.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        x : numpy.ndarray
            An array containing the window centre point genomic positions.
        h12 : numpy.ndarray
            An array with h12 statistic values for each window.

        """
        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = "ag3_h12_gwss_v1"

        params = dict(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._h12_gwss(**params)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        h12 = results["h12"]

        return x, h12

    def _h12_gwss(
        self,
        contig,
        analysis,
        window_size,
        sample_sets,
        sample_query,
        cohort_size,
        random_seed,
    ):

        ds_haps = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()
        pos = ds_haps["variant_position"].values

        h1, h12, h123, h2_h1 = allel.moving_garud_h(ht, size=window_size)

        x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, h12=h12)

        return results

    def plot_h12_gwss_track(
        self,
        contig,
        analysis,
        window_size,
        sample_sets=None,
        sample_query=None,
        cohort_size=30,
        random_seed=42,
        title=None,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=200,
        show=True,
        x_range=None,
    ):
        """Plot h12 GWSS data.

        Parameters
        ----------
        contig: str
            Chromosome arm (e.g., "2L")
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.
        window_size : int
            The size of windows used to calculate h12 over.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.
        width : int, optional
            Plot width in pixels (px).
        height : int. optional
            Plot height in pixels (px).
        show : bool, optional
            If True, show the plot.
        x_range : bokeh.models.Range1d, optional
            X axis range (for linking to other tracks).

        Returns
        -------
        fig : figure
            A plot showing windowed h12 statistic across chosen contig.
        """

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        # compute H12
        x, h12 = self.h12_gwss(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort_size=cohort_size,
            sample_query=sample_query,
            sample_sets=sample_sets,
            random_seed=random_seed,
        )

        # determine X axis range
        x_min = x[0]
        x_max = x[-1]
        if x_range is None:
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        # create a figure
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        if title is None:
            title = sample_query
        fig = bkplt.figure(
            title=title,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            plot_width=width,
            plot_height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, 1),
        )

        # plot H12
        fig.circle(
            x=x,
            y=h12,
            size=3,
            line_width=0.5,
            line_color="black",
            fill_color=None,
        )

        # tidy up the plot
        fig.yaxis.axis_label = "H12"
        fig.yaxis.ticker = [0, 1]
        _bokeh_style_genome_xaxis(fig, contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_h12_gwss(
        self,
        contig,
        analysis,
        window_size,
        sample_sets=None,
        sample_query=None,
        cohort_size=30,
        random_seed=42,
        title=None,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=170,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
    ):
        """Plot h12 GWSS data.

        Parameters
        ----------
        contig: str
            Chromosome arm (e.g., "2L")
        analysis : {"arab", "gamb_colu", "gamb_colu_arab"}
            Which phasing analysis to use. If analysing only An. arabiensis, the
            "arab" analysis is best. If analysing only An. gambiae and An.
            coluzzii, the "gamb_colu" analysis is best. Otherwise, use the
            "gamb_colu_arab" analysis.
        window_size : int
            The size of windows used to calculate h12 over.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.
        width : int, optional
            Plot width in pixels (px).
        track_height : int. optional
            GWSS track height in pixels (px).
        genes_height : int. optional
            Gene track height in pixels (px).

        Returns
        -------
        fig : figure
            A plot showing windowed h12 statistic with gene track on x-axis.
        """

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        # gwss track
        fig1 = self.plot_h12_gwss_track(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            sample_sets=sample_sets,
            sample_query=sample_query,
            cohort_size=cohort_size,
            random_seed=random_seed,
            title=title,
            width=width,
            height=track_height,
            show=False,
        )

        fig1.xaxis.visible = False

        # plot genes
        fig2 = self.plot_genes(
            region=contig,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        # combine plots into a single figure
        fig = bklay.gridplot(
            [fig1, fig2], ncols=1, toolbar_location="above", merge_tools=True
        )

        bkplt.show(fig)


def _roh_hmm_predict(
    *,
    windows,
    counts,
    phet_roh,
    phet_nonroh,
    transition,
    window_size,
    sample_id,
    contig,
):
    # conditional import, pomegranate takes a long time to install on
    # linux due to lack of prebuilt wheels on PyPI
    from allel.stats.misc import tabulate_state_blocks

    # this implementation is based on scikit-allel, but modified to use
    # moving window computation of het counts
    from allel.stats.roh import _hmm_derive_transition_matrix
    from pomegranate import HiddenMarkovModel, PoissonDistribution

    # het probabilities
    het_px = np.concatenate([(phet_roh,), phet_nonroh])

    # start probabilities (all equal)
    start_prob = np.repeat(1 / het_px.size, het_px.size)

    # transition between underlying states
    transition_mx = _hmm_derive_transition_matrix(transition, het_px.size)

    # emission probability distribution
    dists = [PoissonDistribution(x * window_size) for x in het_px]

    # set up model
    # noinspection PyArgumentList
    model = HiddenMarkovModel.from_matrix(
        transition_probabilities=transition_mx, distributions=dists, starts=start_prob
    )

    # predict hidden states
    prediction = np.array(model.predict(counts[:, None]))

    # tabulate runs of homozygosity (state 0)
    # noinspection PyTypeChecker
    df_blocks = tabulate_state_blocks(prediction, states=list(range(len(het_px))))
    df_roh = df_blocks[(df_blocks["state"] == 0)].reset_index(drop=True)

    # adapt the dataframe for ROH
    df_roh["sample_id"] = sample_id
    df_roh["contig"] = contig
    df_roh["roh_start"] = df_roh["start_ridx"].apply(lambda y: windows[y, 0])
    df_roh["roh_stop"] = df_roh["stop_lidx"].apply(lambda y: windows[y, 1])
    df_roh["roh_length"] = df_roh["roh_stop"] - df_roh["roh_start"]
    df_roh.rename(columns={"is_marginal": "roh_is_marginal"}, inplace=True)

    return df_roh[
        [
            "sample_id",
            "contig",
            "roh_start",
            "roh_stop",
            "roh_length",
            "roh_is_marginal",
        ]
    ]


def _setup_taxon_colors(plot_kwargs):
    import plotly.express as px

    taxon_palette = px.colors.qualitative.Plotly
    taxon_color_map = {
        "gambiae": taxon_palette[0],
        "coluzzii": taxon_palette[1],
        "arabiensis": taxon_palette[2],
        "gcx1": taxon_palette[3],
        "gcx2": taxon_palette[4],
        "gcx3": taxon_palette[5],
        "intermediate_gambiae_coluzzii": taxon_palette[6],
        "intermediate_arabiensis_gambiae": taxon_palette[7],
    }
    plot_kwargs.setdefault("color_discrete_map", taxon_color_map)
    plot_kwargs.setdefault("category_orders", {"taxon": list(taxon_color_map.keys())})


def _locate_cohorts(*, cohorts, df_samples):

    # build cohort dictionary where key=cohort_id, value=loc_coh
    coh_dict = {}

    if isinstance(cohorts, dict):
        # user has supplied a custom dictionary mapping cohort identifiers
        # to pandas queries

        for coh, query in cohorts.items():
            # locate samples
            loc_coh = df_samples.eval(query).values
            coh_dict[coh] = loc_coh

    if isinstance(cohorts, str):
        # user has supplied one of the predefined cohort sets

        # fix the string to match columns
        if not cohorts.startswith("cohort_"):
            cohorts = "cohort_" + cohorts

        # check the given cohort set exists
        if cohorts not in df_samples.columns:
            raise ValueError(f"{cohorts!r} is not a known cohort set")
        cohort_labels = df_samples[cohorts].unique()

        # remove the nans and sort
        cohort_labels = sorted([c for c in cohort_labels if isinstance(c, str)])
        for coh in cohort_labels:
            loc_coh = df_samples[cohorts] == coh
            coh_dict[coh] = loc_coh.values

    return coh_dict


@numba.njit("Tuple((int8, int64))(int8[:], int8)")
def _cn_mode_1d(a, vmax):

    # setup intermediates
    m = a.shape[0]
    counts = np.zeros(vmax + 1, dtype=numba.int64)

    # initialise return values
    mode = numba.int8(-1)
    mode_count = numba.int64(0)

    # iterate over array values, keeping track of counts
    for i in range(m):
        v = a[i]
        if 0 <= v <= vmax:
            c = counts[v]
            c += 1
            counts[v] = c
            if c > mode_count:
                mode = v
                mode_count = c
            elif c == mode_count and v < mode:
                # consistency with scipy.stats, break ties by taking lower value
                mode = v

    return mode, mode_count


@numba.njit("Tuple((int8[:], int64[:]))(int8[:, :], int8)")
def _cn_mode(a, vmax):

    # setup intermediates
    n = a.shape[1]

    # setup outputs
    modes = np.zeros(n, dtype=numba.int8)
    counts = np.zeros(n, dtype=numba.int64)

    # iterate over columns, computing modes
    for j in range(a.shape[1]):
        mode, count = _cn_mode_1d(a[:, j], vmax)
        modes[j] = mode
        counts[j] = count

    return modes, counts


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
        return pd.Period(freq="A", year=year)
    else:
        return pd.NaT


@numba.njit
def _cohort_alt_allele_counts_melt_kernel(gt, indices, max_allele):

    n_variants = gt.shape[0]
    n_indices = indices.shape[0]
    ploidy = gt.shape[2]

    ac_alt_melt = np.zeros(n_variants * max_allele, dtype=np.int64)
    an = np.zeros(n_variants, dtype=np.int64)

    for i in range(n_variants):
        out_i_offset = (i * max_allele) - 1
        for j in range(n_indices):
            ix = indices[j]
            for k in range(ploidy):
                allele = gt[i, ix, k]
                if allele > 0:
                    out_i = out_i_offset + allele
                    ac_alt_melt[out_i] += 1
                    an[i] += 1
                elif allele == 0:
                    an[i] += 1

    return ac_alt_melt, an


def _cohort_alt_allele_counts_melt(gt, indices, max_allele):
    ac_alt_melt, an = _cohort_alt_allele_counts_melt_kernel(gt, indices, max_allele)
    an_melt = np.repeat(an, max_allele, axis=0)
    return ac_alt_melt, an_melt


def _make_snp_label(contig, position, ref_allele, alt_allele):
    return f"{contig}:{position:,} {ref_allele}>{alt_allele}"


def _make_snp_label_effect(contig, position, ref_allele, alt_allele, aa_change):
    label = f"{contig}:{position:,} {ref_allele}>{alt_allele}"
    if isinstance(aa_change, str):
        label += f" ({aa_change})"
    return label


def _make_snp_label_aa(aa_change, contig, position, ref_allele, alt_allele):
    label = f"{aa_change} ({contig}:{position:,} {ref_allele}>{alt_allele})"
    return label


def _make_gene_cnv_label(gene_id, gene_name, cnv_type):
    label = gene_id
    if isinstance(gene_name, str):
        label += f" ({gene_name})"
    label += f" {cnv_type}"
    return label


def _map_snp_to_aa_change_frq_ds(ds):

    # keep only variables that make sense for amino acid substitutions
    keep_vars = [
        "variant_contig",
        "variant_position",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_aa_pos",
        "variant_aa_change",
        "variant_ref_allele",
        "variant_ref_aa",
        "variant_alt_aa",
        "event_nobs",
    ]

    if ds.dims["variants"] == 1:

        # keep everything as-is, no need for aggregation
        ds_out = ds[keep_vars + ["variant_alt_allele", "event_count"]]

    else:

        # take the first value from all variants variables
        ds_out = ds[keep_vars].isel(variants=[0])

        # sum event count over variants
        count = ds["event_count"].values.sum(axis=0, keepdims=True)
        ds_out["event_count"] = ("variants", "cohorts"), count

        # collapse alt allele
        alt_allele = "{" + ",".join(ds["variant_alt_allele"].values) + "}"
        ds_out["variant_alt_allele"] = "variants", np.array([alt_allele], dtype=object)

    return ds_out


def _add_frequency_ci(ds, ci_method):
    from statsmodels.stats.proportion import proportion_confint

    if ci_method is not None:
        count = ds["event_count"].values
        nobs = ds["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frq_ci_low, frq_ci_upp = proportion_confint(
                count=count, nobs=nobs, method=ci_method
            )
        ds["event_frequency_ci_low"] = ("variants", "cohorts"), frq_ci_low
        ds["event_frequency_ci_upp"] = ("variants", "cohorts"), frq_ci_upp


def _prep_samples_for_cohort_grouping(*, df_samples, area_by, period_by):

    # take a copy, as we will modify the dataframe
    df_samples = df_samples.copy()

    # fix intermediate taxon values - we only want to build cohorts with clean
    # taxon calls, so we set intermediate values to None
    loc_intermediate_taxon = (
        df_samples["taxon"].str.startswith("intermediate").fillna(False)
    )
    df_samples.loc[loc_intermediate_taxon, "taxon"] = None

    # add period column
    if period_by == "year":
        make_period = _make_sample_period_year
    elif period_by == "quarter":
        make_period = _make_sample_period_quarter
    elif period_by == "month":
        make_period = _make_sample_period_month
    else:
        raise ValueError(
            f"Value for period_by parameter must be one of 'year', 'quarter', 'month'; found {period_by!r}."
        )
    sample_period = df_samples.apply(make_period, axis="columns")
    df_samples["period"] = sample_period

    # add area column for consistent output
    df_samples["area"] = df_samples[area_by]

    return df_samples


def _build_cohorts_from_sample_grouping(group_samples_by_cohort, min_cohort_size):

    # build cohorts dataframe
    df_cohorts = group_samples_by_cohort.agg(
        size=("sample_id", len),
        lat_mean=("latitude", "mean"),
        lat_max=("latitude", "mean"),
        lat_min=("latitude", "mean"),
        lon_mean=("longitude", "mean"),
        lon_max=("longitude", "mean"),
        lon_min=("longitude", "mean"),
    )
    # reset index so that the index fields are included as columns
    df_cohorts = df_cohorts.reset_index()

    # add cohort helper variables
    cohort_period_start = df_cohorts["period"].apply(lambda v: v.start_time)
    cohort_period_end = df_cohorts["period"].apply(lambda v: v.end_time)
    df_cohorts["period_start"] = cohort_period_start
    df_cohorts["period_end"] = cohort_period_end
    # create a label that is similar to the cohort metadata,
    # although this won't be perfect
    df_cohorts["label"] = df_cohorts.apply(
        lambda v: f"{v.area}_{v.taxon[:4]}_{v.period}", axis="columns"
    )

    # apply minimum cohort size
    df_cohorts = df_cohorts.query(f"size >= {min_cohort_size}").reset_index(drop=True)

    return df_cohorts


def _check_param_min_cohort_size(min_cohort_size):
    if not isinstance(min_cohort_size, int):
        raise TypeError(
            f"Type of parameter min_cohort_size must be int; found {type(min_cohort_size)}."
        )
    if min_cohort_size < 1:
        raise ValueError(
            f"Value of parameter min_cohort_size must be at least 1; found {min_cohort_size}."
        )


def _pandas_apply(f, df, columns):
    """Optimised alternative to pandas apply."""
    df = df.reset_index(drop=True)
    iterator = zip(*[df[c].values for c in columns])
    ret = pd.Series((f(*vals) for vals in iterator))
    return ret


def _region_str(region):
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
            return "; ".join([_region_str(r) for r in region])
        else:
            region = region[0]

    # sanity check
    assert isinstance(region, Region)

    return str(region)


def _bokeh_style_genome_xaxis(fig, contig):
    """Standard styling for X axis of genome plots."""
    import bokeh.models as bkmod

    fig.xaxis.axis_label = f"Contig {contig} position (bp)"
    fig.xaxis.ticker = bkmod.AdaptiveTicker(min_interval=1)
    fig.xaxis.minor_tick_line_color = None
    fig.xaxis[0].formatter = bkmod.NumeralTickFormatter(format="0,0")
