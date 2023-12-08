import json
from contextlib import nullcontext
from pathlib import Path
from typing import (
    IO,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import bokeh.io
import numpy as np
import pandas as pd
import zarr
from numpydoc_decorator import doc
from tqdm.auto import tqdm as tqdm_auto
from tqdm.dask import TqdmCallback
from yaspin import yaspin

from ..util import (
    CacheMiss,
    LoggingHelper,
    check_colab_location,
    check_types,
    hash_params,
    init_filesystem,
)
from . import base_params


class AnophelesBase:
    def __init__(
        self,
        *,
        url: str,
        config_path: str,
        pre: bool,
        gcs_url: Optional[str],  # only used for colab location check
        major_version_number: int,
        major_version_path: str,
        bokeh_output_notebook: bool = False,
        log: Optional[Union[str, IO]] = None,
        debug: bool = False,
        show_progress: bool = False,
        check_location: bool = False,
        storage_options: Optional[Mapping] = None,
        results_cache: Optional[str] = None,
        tqdm_class=None,
    ):
        self._url = url
        self._config_path = config_path
        self._pre = pre
        self._gcs_url = gcs_url
        self._major_version_number = major_version_number
        self._major_version_path = major_version_path
        self._debug = debug
        self._show_progress = show_progress
        if tqdm_class is None:
            tqdm_class = tqdm_auto
        self._tqdm_class = tqdm_class

        # Set up logging.
        self._log = LoggingHelper(name=__name__, out=log, debug=debug)

        # Set up fsspec filesystem. N.B., we use fsspec here to allow for
        # accessing different types of storage - fsspec will automatically
        # detect which type of storage to use based on the URL provided.
        # E.g., if the URL begins with "gs://" then a GCSFileSystem will
        # be used to read from Google Cloud Storage.
        if storage_options is None:
            storage_options = dict()
        self._fs, self._base_path = init_filesystem(url, **storage_options)

        # Lazily load config.
        self._config: Optional[Dict] = None

        # Get bokeh to output plots to the notebook - this is a common gotcha,
        # users forget to do this and wonder why bokeh plots don't show.
        if bokeh_output_notebook:  # pragma: no cover
            bokeh.io.output_notebook(hide_banner=True)

        # Check colab location is in the US.
        if check_location and self._gcs_url is not None:  # pragma: no cover
            self._client_details = check_colab_location(
                gcs_url=self._gcs_url, url=self._url
            )
        else:
            self._client_details = None

        # Set up cache attributes.
        self._cache_releases: Optional[Tuple[str, ...]] = None
        self._cache_sample_sets: Dict[str, pd.DataFrame] = dict()
        self._cache_sample_set_to_release: Optional[Dict[str, str]] = None
        self._cache_sample_set_to_study: Optional[Dict[str, str]] = None
        self._cache_files: Dict[str, bytes] = dict()

        # Set up results cache directory path.
        self._results_cache: Optional[Path] = None
        if results_cache is not None:
            self._results_cache = Path(results_cache).expanduser().resolve()

    def _progress(self, iterable, desc=None, leave=False, **kwargs):  # pragma: no cover
        # Progress doesn't mix well with debug logging.
        show_progress = self._show_progress and not self._debug
        if show_progress:
            return self._tqdm_class(iterable, desc=desc, leave=leave, **kwargs)
        else:
            return iterable

    def _dask_progress(self, desc=None, leave=False, **kwargs):  # pragma: no cover
        # Progress doesn't mix well with debug logging.
        show_progress = self._show_progress and not self._debug
        if show_progress:
            return TqdmCallback(
                desc=desc, leave=leave, tqdm_class=self._tqdm_class, **kwargs
            )
        else:
            return nullcontext()

    def _spinner(
        self, desc=None, spinner=None, side="right", timer=True, **kwargs
    ):  # pragma: no cover
        # Progress doesn't mix well with debug logging.
        show_progress = self._show_progress and not self._debug
        if show_progress:
            if desc:
                # For consistent behaviour with tqdm.
                desc += ":"
            return yaspin(text=desc, spinner=spinner, side=side, timer=timer, **kwargs)
        else:
            return nullcontext()

    @check_types
    def open_file(self, path: str) -> IO:
        full_path = f"{self._base_path}/{path}"
        return self._fs.open(full_path)

    @check_types
    def read_files(
        self,
        paths: Iterable[str],
        on_error: Literal["raise", "omit", "return"] = "return",
    ) -> Mapping[str, Union[bytes, Exception]]:
        # Check for any cached files.
        files = {
            path: data for path, data in self._cache_files.items() if path in paths
        }
        paths_not_cached = [p for p in paths if p not in self._cache_files]

        if paths_not_cached:
            # Prepend the base path.
            prefix = self._base_path + "/"
            full_paths = [prefix + path for path in paths_not_cached]

            # Retrieve all files. N.B., depending on what type of storage is
            # being used, the cat() function may be able to read multiple files
            # concurrently. E.g., this is true if the file system is a
            # GCSFileSystem, which achieves concurrency by using async I/O under the
            # hood. This is a useful performance optimisation, because reading a
            # file from GCS incurs some latency, and so reading many small files
            # from GCS can be slow if files are not read concurrently. Hence we
            # want to make use of cat() here and provide paths for all files to
            # read concurrently. For more information see:
            # https://filesystem-spec.readthedocs.io/en/latest/async.html
            full_path_files = self._fs.cat(full_paths, on_error=on_error)

            # Strip off the prefix.
            retrieved_files = {
                k.split(prefix, 1)[1]: v for k, v in full_path_files.items()
            }

            # Update the cache.
            self._cache_files.update(retrieved_files)

            # Add retrieved files to the result.
            files.update(retrieved_files)

        return files

    @property
    def config(self) -> Dict:
        if self._config is None:
            with self.open_file(self._config_path) as f:
                self._config = json.load(f)
        return self._config.copy()

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

    @doc(
        summary="""
            Compatibility function, allows us to use release identifiers like "3.0"
            and "3.1" in the public API, and map these internally into storage path
            segments.
        """,
    )
    def _release_to_path(self, release: base_params.release) -> str:
        if release == f"{self._major_version_number}.0":
            # special case
            return self._major_version_path
        elif isinstance(release, str) and release.startswith(
            f"{self._major_version_number}."
        ):
            return f"v{release}"
        else:
            raise ValueError(f"Invalid release: {release!r}")

    @doc(
        summary="""
            Compatibility function, allows us to use release identifiers like "3.0"
            and "3.1" in the public API, and map these internally into storage path
            segments.
        """,
        parameters=dict(
            path="Path segment.",
        ),
    )
    def _path_to_release(self, path: str) -> str:
        if path == self._major_version_path:
            return f"{self._major_version_number}.0"
        elif path.startswith(f"v{self._major_version_number}."):
            return path[1:]
        else:
            raise RuntimeError(f"Unexpected release path: {path!r}")

    def _public_releases(self) -> Tuple[str, ...]:
        releases = tuple(self.config.get("PUBLIC_RELEASES", ()))
        return releases

    @doc(
        summary="""
            Here we discover which releases are available, by listing the storage
            directory and examining the subdirectories. This may include "pre-releases"
            where data may be incomplete.
        """
    )
    def _discover_releases(self) -> Tuple[str, ...]:
        sub_dirs = sorted([p.split("/")[-1] for p in self._fs.ls(self._base_path)])
        discovered_releases = tuple(
            sorted(
                [
                    self._path_to_release(d)
                    for d in sub_dirs
                    # FIXME: this matches v3 and v3.1, but also v3001.1
                    if d.startswith(f"v{self._major_version_number}")
                    and self._fs.exists(f"{self._base_path}/{d}/manifest.tsv")
                ]
            )
        )
        return discovered_releases

    @property
    def releases(self) -> Tuple[str, ...]:
        """Currently available data releases."""
        if self._cache_releases is None:
            if self._pre:
                self._cache_releases = self._discover_releases()
            else:
                self._cache_releases = self._public_releases()
        return self._cache_releases

    @property
    def client_location(self) -> str:
        details = self._client_details
        if details is not None:
            region = details.region
            country = details.country
            location = f"{region}, {country}"
        else:
            location = "unknown"
        return location

    def _read_sample_sets(self, *, single_release: str):
        """Read the manifest of sample sets for a single release."""
        # Construct a path for the manifest file.
        release_path = self._release_to_path(single_release)
        manifest_path = f"{release_path}/manifest.tsv"

        # Read the manifest into a pandas dataframe.
        with self.open_file(manifest_path) as f:
            df = pd.read_csv(f, sep="\t", na_values="")

        # Add a "release" column for convenience.
        df["release"] = single_release
        return df

    @check_types
    @doc(
        summary="Access a dataframe of sample sets",
        returns="A dataframe of sample sets, one row per sample set.",
    )
    def sample_sets(
        self,
        release: Optional[base_params.release] = None,
    ) -> pd.DataFrame:
        if release is None:
            # Retrieve sample sets from all available releases.
            release = self.releases

        if isinstance(release, str):
            # Retrieve sample sets for a single release.

            if release not in self.releases:
                raise ValueError(f"Release not available: {release!r}")

            try:
                df = self._cache_sample_sets[release]

            except KeyError:
                # Read and cache dataframe for performance.
                df = self._read_sample_sets(single_release=release)
                self._cache_sample_sets[release] = df

        elif isinstance(release, Sequence):
            # Ensure no duplicates.
            releases = sorted(set(release))

            # Retrieve and concatenate sample sets from multiple releases.
            df = pd.concat(
                [self.sample_sets(release=r) for r in releases],
                axis=0,
                ignore_index=True,
            )

        else:
            raise TypeError

        # Return copy to ensure cached dataframes aren't modified by user.
        return df.copy()

    @check_types
    @doc(
        summary="Find which release a sample set was included in.",
    )
    def lookup_release(self, sample_set: base_params.sample_set):
        if self._cache_sample_set_to_release is None:
            df_sample_sets = self.sample_sets().set_index("sample_set")
            self._cache_sample_set_to_release = df_sample_sets["release"].to_dict()

        try:
            return self._cache_sample_set_to_release[sample_set]
        except KeyError:
            raise ValueError(f"No release found for sample set {sample_set!r}")

    @check_types
    @doc(
        summary="Find which study a sample set belongs to.",
    )
    def lookup_study(self, sample_set: base_params.sample_set):
        if self._cache_sample_set_to_study is None:
            df_sample_sets = self.sample_sets().set_index("sample_set")
            self._cache_sample_set_to_study = df_sample_sets["study_id"].to_dict()
        try:
            return self._cache_sample_set_to_study[sample_set]
        except KeyError:
            raise ValueError(f"No study ID found for sample set {sample_set!r}")

    def _prep_sample_sets_param(
        self, *, sample_sets: Optional[base_params.sample_sets]
    ) -> List[str]:
        """Common handling for the `sample_sets` parameter. For convenience, we
        allow this to be a single sample set, or a list of sample sets, or a
        release identifier, or a list of release identifiers."""

        all_sample_sets = self.sample_sets()["sample_set"].to_list()

        if sample_sets is None:
            # All available sample sets.
            prepped_sample_sets = all_sample_sets

        elif isinstance(sample_sets, str):
            if sample_sets.startswith(f"{self._major_version_number}."):
                # Convenience, can use a release identifier to denote all sample sets in a release.
                prepped_sample_sets = self.sample_sets(release=sample_sets)[
                    "sample_set"
                ].to_list()

            else:
                # Single sample set, normalise to always return a list.
                prepped_sample_sets = [sample_sets]

        else:
            # Sequence of sample sets or releases.
            assert isinstance(sample_sets, Sequence)
            prepped_sample_sets = []
            for s in sample_sets:
                # Make a recursive call to handle the case where s is a release identifier.
                sp = self._prep_sample_sets_param(sample_sets=s)

                # Make sure we end up with a flat list of sample sets.
                prepped_sample_sets.extend(sp)

        # Ensure all sample sets selected at most once.
        prepped_sample_sets = sorted(set(prepped_sample_sets))

        # Check for bad sample sets.
        for s in prepped_sample_sets:
            if s not in all_sample_sets:
                raise ValueError(f"Sample set {s!r} not found.")

        return prepped_sample_sets

    def _results_cache_add_analysis_params(self, params: dict):
        # Expect sub-classes will override to add any analysis parameters.
        pass

    @check_types
    def results_cache_get(
        self, *, name: str, params: Dict[str, Any]
    ) -> Mapping[str, np.ndarray]:
        name = type(self).__name__.lower() + "_" + name
        if self._results_cache is None:
            raise CacheMiss
        params = params.copy()
        self._results_cache_add_analysis_params(params)
        cache_key, _ = hash_params(params)
        cache_path = self._results_cache / name / cache_key

        # Read zipped zarr format.
        results_path = cache_path / "results.zarr.zip"
        if results_path.exists():
            return zarr.load(results_path)

        # For backwards compatibility, read npz format.
        legacy_results_path = cache_path / "results.npz"
        if legacy_results_path.exists():  # pragma: no cover
            return np.load(legacy_results_path)

        raise CacheMiss

    @check_types
    def results_cache_set(
        self, *, name: str, params: Dict[str, Any], results: Mapping[str, np.ndarray]
    ):
        name = type(self).__name__.lower() + "_" + name
        if self._results_cache is None:
            return

        # Set up parameters for the results to be saved.
        params = params.copy()
        self._results_cache_add_analysis_params(params)
        cache_key, params_json = hash_params(params)

        # Determine storage path.
        cache_path = self._results_cache / name / cache_key
        cache_path.mkdir(exist_ok=True, parents=True)

        # Write the parameters as a JSON file.
        params_path = cache_path / "params.json"

        # Write the data to be cached as a zipped zarr file.
        results_path = cache_path / "results.zarr.zip"

        with self._spinner("Save results to cache"):
            with params_path.open(mode="w") as f:
                f.write(params_json)
            zarr.save(results_path, **results)
