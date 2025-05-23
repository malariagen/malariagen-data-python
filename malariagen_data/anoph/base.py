import os

import json
from contextlib import nullcontext
from datetime import date
from pathlib import Path
import re
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
from textwrap import dedent
import bokeh.io
import ipinfo  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import zarr  # type: ignore
from numpydoc_decorator import doc  # type: ignore
from tqdm.auto import tqdm as tqdm_auto  # type: ignore
from tqdm.dask import TqdmCallback  # type: ignore
from yaspin import yaspin  # type: ignore

from ..util import (
    CacheMiss,
    LoggingHelper,
    check_colab_location,
    check_types,
    distributed_client,
    get_gcp_region,
    hash_params,
    init_filesystem,
)
from . import base_params


class AnophelesBase:
    def __init__(
        self,
        *,
        url: str,
        public_url: str,
        config_path: str,
        pre: bool,
        major_version_number: int,
        major_version_path: str,
        gcs_default_url: Optional[str] = None,
        gcs_region_urls: Mapping[str, str] = {},
        bokeh_output_notebook: bool = False,
        log: Optional[Union[str, IO]] = None,
        debug: bool = False,
        show_progress: Optional[bool] = None,
        check_location: bool = False,
        storage_options: Optional[Mapping] = None,
        results_cache: Optional[str] = None,
        tqdm_class=None,
        unrestricted_use_only: Optional[bool] = False,
        surveillance_use_only: Optional[bool] = False,
    ):
        # If show_progress has not been specified, then determine the default.
        if show_progress is None:
            # Get the env var, if it exists.
            show_progress_env = os.getenv("MGEN_SHOW_PROGRESS")

            # If the env var does not exist, then use the class default.
            # Otherwise, convert the env var value to a boolean and use that.
            if show_progress_env is None:
                show_progress = True
            else:
                show_progress = show_progress_env.lower() in ("true", "1", "yes", "on")

        self._config_path = config_path
        self._pre = pre
        self._gcs_default_url = gcs_default_url
        self._gcs_region_urls = gcs_region_urls
        self._major_version_number = major_version_number
        self._major_version_path = major_version_path
        self._debug = debug
        self._show_progress = show_progress
        if tqdm_class is None:
            tqdm_class = tqdm_auto
        self._tqdm_class = tqdm_class
        self._unrestricted_use_only = unrestricted_use_only
        self._surveillance_use_only = surveillance_use_only

        # Set up logging.
        self._log = LoggingHelper(name=__name__, out=log, debug=debug)

        # Check client location.
        self._client_details = None
        if check_location:
            try:
                self._client_details = ipinfo.getHandler().getDetails()
            except OSError:
                pass

        # Determine cloud location details.
        self._gcp_region = get_gcp_region(self._client_details)

        # Check colab location.
        check_colab_location(self._gcp_region)

        # Determine storage URL.
        if url:
            # User has explicitly provided a URL to use.
            self._url = url
        elif self._gcp_region in self._gcs_region_urls:
            # Choose URL in the same GCP region.
            self._url = self._gcs_region_urls[self._gcp_region]
        elif self._gcs_default_url:
            # Fall back to default URL if available.
            self._url = self._gcs_default_url
        else:
            raise ValueError("A value for the `url` parameter must be provided.")
        del url

        self._public_url = public_url

        # Set up fsspec filesystem. N.B., we use fsspec here to allow for
        # accessing different types of storage - fsspec will automatically
        # detect which type of storage to use based on the URL provided.
        # E.g., if the URL begins with "gs://" then a GCSFileSystem will
        # be used to read from Google Cloud Storage.
        if storage_options is None:
            storage_options = dict()
        try:
            self._fs, self._base_path = init_filesystem(self._url, **storage_options)
        except Exception as exc:  # pragma: no cover
            raise IOError(
                "An error occurred establishing a connection to the storage system. Please see the nested exception for more details."
            ) from exc

        # Eagerly load config to trigger any access problems early.
        try:
            with self.open_file(self._config_path) as f:
                self._config = json.load(f)
        except Exception as exc:  # pragma: no cover
            if (isinstance(exc, OSError) and "forbidden" in str(exc).lower()) or (
                getattr(exc, "status", None) == 403
            ):
                # This seems to be the best way to detect the case where the
                # current user is trying to access GCS but has not been granted
                # permissions. Reraise with a helpful message.
                raise PermissionError(
                    dedent(
                        """
                           Your Google account does not appear to have permission to access the data.
                           If you have not yet submitted a data access request, please complete the form
                           at the following link: https://forms.gle/d1NV3aL3EoVQGSHYA

                           If you are still experiencing problems accessing data, please email
                           support@malariagen.net for assistance.
                        """
                    )
                ) from exc
            else:
                # Some other kind of error, reraise.
                raise exc

        # Get bokeh to output plots to the notebook - this is a common gotcha,
        # users forget to do this and wonder why bokeh plots don't show.
        if bokeh_output_notebook:  # pragma: no cover
            bokeh.io.output_notebook(hide_banner=True)

        # Set up cache attributes.
        self._cache_releases: Optional[Tuple[str, ...]] = None
        self._cache_available_releases: Optional[Tuple[str, ...]] = None
        self._cache_sample_sets: Dict[str, pd.DataFrame] = dict()
        self._cache_available_sample_sets: Dict[str, pd.DataFrame] = dict()
        self._cache_sample_set_to_release: Optional[Dict[str, str]] = None
        self._cache_sample_set_to_study: Optional[Dict[str, str]] = None
        self._cache_sample_set_to_study_info: Optional[Dict[str, dict]] = None
        self._cache_sample_set_to_terms_of_use_info: Optional[Dict[str, dict]] = None
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
            if distributed_client():
                # Cannot easily show progress, fall back to spinner.
                return self._spinner(desc=desc)
            else:
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
        sub_dirs = sorted(
            [p.split("/")[-1] for p in self._fs.ls(self._base_path, detail=False)]
        )
        # Note: this matches v3, v3. and v3.1, but not v3001.1
        version_pattern = re.compile(f"^v{self._major_version_number}(\\..*)?$")
        # To sort the versions numerically, we use a lambda function for the "key" parameter of sorted().
        # The lambda function splits each version string into a list of its integer parts, using split('.') and int(), e.g. [3, 1],
        # which sorted() then uses to determine the order, as opposed to the default lexicographic order.
        discovered_releases = tuple(
            sorted(
                [
                    self._path_to_release(d)
                    for d in sub_dirs
                    if version_pattern.match(d)
                    and self._fs.exists(f"{self._base_path}/{d}/manifest.tsv")
                ],
                key=lambda v: [int(part) for part in v.split(".")],
            )
        )
        return discovered_releases

    @property
    def _available_releases(self) -> Tuple[str, ...]:
        """Currently available data releases, regardless of `unrestricted_use_only` and `surveillance_use_only`. When `pre` is set to `True`, this includes "pre-releases", otherwise only the "public" releases."""
        if self._cache_available_releases is None:
            if self._pre:
                self._cache_available_releases = self._discover_releases()
            else:
                self._cache_available_releases = self._public_releases()

        return self._cache_available_releases

    @property
    def _releases_with_unrestricted_data(self) -> Tuple[str, ...]:
        """Releases that contain some unrestricted data."""

        # Start a list of releases that contain some unrestricted data.
        releases_with_unrestricted_data = []

        # Get the available releases, which depends on the `pre` setting.
        available_releases = self._available_releases

        # For each available release...
        for release in available_releases:
            # Determine whether this release contains any unrestricted data.
            if self._release_has_unrestricted_data(release=release):
                releases_with_unrestricted_data.append(release)

        return tuple(releases_with_unrestricted_data)

    @property
    def _releases_with_surveillance_data(self) -> Tuple[str, ...]:
        """Releases that contain some surveillance data."""

        # Start a list of releases that contain some surveillance data.
        releases_with_surveillance_data = []

        # Get the available releases, which will depend on the `pre` setting.
        available_releases = self._available_releases

        # For each available release...
        for release in available_releases:
            # Determine whether this release contains any surveillance data.
            if self._release_has_surveillance_data(release=release):
                releases_with_surveillance_data.append(release)

        return tuple(releases_with_surveillance_data)

    @property
    def _relevant_releases(self) -> Tuple[str, ...]:
        """Relevant data releases. When `unrestricted_use_only` is set to `True`, only releases that contain some unrestricted data will be included. When `surveillance_use_only` is set to true, only releases that contain some surveillance data will be included. When `pre` is set to `True`, this includes "pre-releases", otherwise only the "public" releases."""

        if self._cache_releases is None:
            # Start a list of the relevant releases.
            relevant_releases = []  # type: List[str]

            # Get the available releases, which depends on the `pre` setting.
            available_releases = self._available_releases

            # If there are no criteria, then all available releases are relevant.
            if not self._unrestricted_use_only and not self._surveillance_use_only:
                relevant_releases = list(available_releases)

            elif self._unrestricted_use_only and not self._surveillance_use_only:
                # Get the releases with unrestricted data.
                releases_with_unrestricted_data = self._releases_with_unrestricted_data

                # Determine whether each release is relevant to the specified criteria.
                for release in available_releases:
                    # Determine whether this release has any unrestricted data.
                    has_unrestricted_data = release in releases_with_unrestricted_data

                    # If we want unrestricted data, but this release doesn't have any, then don't include it.
                    if self._unrestricted_use_only and not has_unrestricted_data:
                        continue

                    # Otherwise, this release is relevant, so include it.
                    relevant_releases.append(release)

            elif not self._unrestricted_use_only and self._surveillance_use_only:
                # Get the releases with surveillance data.
                releases_with_surveillance_data = self._releases_with_surveillance_data

                # Determine whether each release is relevant to the specified criteria.
                for release in available_releases:
                    # Determine whether this release has any surveillance data.
                    has_surveillance_data = release in releases_with_surveillance_data

                    # If we want surveillance data, but this release doesn't have any, then don't include it.
                    if self._surveillance_use_only and not has_surveillance_data:
                        continue

                    # Otherwise, this release is relevant, so include it.
                    relevant_releases.append(release)

            elif self._unrestricted_use_only and self._surveillance_use_only:
                # Get the releases with unrestricted data.
                releases_with_unrestricted_data = self._releases_with_unrestricted_data

                # Get the releases with surveillance data.
                releases_with_surveillance_data = self._releases_with_surveillance_data

                # Determine whether each release is relevant to the specified criteria.
                for release in available_releases:
                    # Determine whether this release has any unrestricted data.
                    has_unrestricted_data = release in releases_with_unrestricted_data

                    # Determine whether this release has any surveillance data.
                    has_surveillance_data = release in releases_with_surveillance_data

                    # If we want unrestricted data, but this release doesn't have any, then don't include it.
                    if self._unrestricted_use_only and not has_unrestricted_data:
                        continue

                    # If we want surveillance data, but this release doesn't have any, then don't include it.
                    if self._surveillance_use_only and not has_surveillance_data:
                        continue

                    # Otherwise, this release is relevant, so include it.
                    relevant_releases.append(release)

            self._cache_releases = tuple(relevant_releases)

        return self._cache_releases

    @property
    def releases(self) -> Tuple[str, ...]:
        """Relevant data releases. When `unrestricted_use_only` is set to `True`, only releases that contain some unrestricted data will be included. When `surveillance_use_only` is set to true, only releases that contain some surveillance data will be included. When `pre` is set to `True`, this includes "pre-releases", otherwise only the "public" releases."""
        return self._relevant_releases

    @property
    def client_location(self) -> str:
        details = self._client_details
        if details is not None:
            region = details.region
            country = details.country_name
            location = f"{region}, {country}"
            if self._gcp_region:
                location += f" (Google Cloud {self._gcp_region})"
        else:
            location = "unknown"
        return location

    def _surveillance_flags(self, sample_sets: List[str]):
        raise NotImplementedError("Subclasses must implement `_surveillance_flags`.")

    def _release_has_unrestricted_data(self, *, release: str):
        """Return `True` if the specified release has any unrestricted data. Otherwise return `False`."""

        # The release has unrestricted data if any of its sample sets are marked as unrestricted.
        # `_read_sample_sets_manifest` gives the sample sets manifest for a release as a DataFrame, potentially with an `unrestricted_use` column.

        # Get the sample sets manifest for the specified release, potentially with the derived `unrestricted_use` column.
        sample_sets_manifest_df = self._read_sample_sets_manifest(
            single_release=release
        )

        # Determine whether any of the sample sets in the manifest are marked as unrestricted.
        release_has_unrestricted_data = (
            "unrestricted_use" in sample_sets_manifest_df.columns
            and sample_sets_manifest_df["unrestricted_use"].any()
        )

        return release_has_unrestricted_data

    def _release_has_surveillance_data(self, *, release: str):
        """Return `True` if the specified release has any surveillance data. Otherwise return `False`."""

        # The release has surveillance data if any of its sample sets have any samples that are flagged as `is_surveillance`.

        # Get the list of sample sets for the specified release.
        # Note: rather than using `sample_sets()`, to avoid additional processing, we are using `_read_sample_sets_manifest()`.
        sample_sets_manifest_df = self._read_sample_sets_manifest(
            single_release=release
        )
        sample_sets = sample_sets_manifest_df["sample_set"].to_list()

        # Determine whether any of the sample sets have surveillance data.
        release_has_surveillance_data = False
        for sample_set in sample_sets:
            if self._sample_set_has_surveillance_data(sample_set=sample_set):
                release_has_surveillance_data = True
                break

        return release_has_surveillance_data

    def _sample_set_has_surveillance_data(self, *, sample_set: str):
        """Return `True` if the specified sample set has any surveillance data. Otherwise return `False`."""

        # Get the surveillance flags for this sample set.
        sample_set_surveillance_flags_df = self._surveillance_flags(
            sample_sets=[sample_set]
        )

        # Determine whether there are any samples in this sample set with `is_surveillance` set to `True`.
        sample_set_has_surveillance_data = (
            "is_surveillance" in sample_set_surveillance_flags_df.columns
            and sample_set_surveillance_flags_df["is_surveillance"].any()
        )

        return sample_set_has_surveillance_data

    def _sample_set_has_unrestricted_use(self, *, sample_set: str):
        """Return `True` if the specified sample set has any unrestricted use. Otherwise return `False`."""

        # Get the manifest data for this sample set.
        sample_set_release = self.lookup_release(sample_set)
        release_manifest_df = self._read_sample_sets_manifest(
            single_release=sample_set_release
        )
        sample_set_records_srs = release_manifest_df.loc[
            release_manifest_df["sample_set"] == sample_set, "unrestricted_use"
        ]

        if len(sample_set_records_srs) == 0:
            raise ValueError(
                f"No release manifest info found for sample_set '{sample_set}'"
            )
        elif len(sample_set_records_srs) > 1:
            raise ValueError(
                f"More than one record found in the release manifest for sample_set '{sample_set}'"
            )
        else:
            # Convert the NumPy boolean to a standard Python bool.
            sample_set_has_unrestricted_use = bool(sample_set_records_srs.iloc[0])

        return sample_set_has_unrestricted_use

    def _read_sample_sets_manifest(self, *, single_release: str):
        """Read the manifest of sample sets for a single release."""
        # Construct a path for the manifest file.
        release_path = self._release_to_path(single_release)
        manifest_path = f"{release_path}/manifest.tsv"

        # Read the manifest into a pandas dataframe.
        with self.open_file(manifest_path) as f:
            df = pd.read_csv(f, sep="\t", na_values="")

        # Add a "release" column for convenience.
        df["release"] = single_release

        # Note: terms-of-use columns might not exist in the manifest, e.g. during pre-release.
        # If there is a terms-of-use expiry date, derive the "unrestricted_use".
        terms_of_use_expiry_date_column = "terms_of_use_expiry_date"
        if terms_of_use_expiry_date_column in df.columns:
            # Get today's date in ISO format
            today_date_iso = date.today().isoformat()
            # Add an "unrestricted_use" column, set to True if terms-of-use expiry date <= today's date.
            df["unrestricted_use"] = df[terms_of_use_expiry_date_column].apply(
                lambda d: True if pd.isna(d) else (d <= today_date_iso)
            )
            # Make the "unrestricted_use" column a nullable boolean, to allow missing data.
            df["unrestricted_use"] = df["unrestricted_use"].astype(pd.BooleanDtype())

        return df

    @check_types
    @doc(
        summary="Access a dataframe of sample sets",
        returns="""A dataframe of sample sets, one row per sample set. It contains five columns:
         `sample_set` is the name of the sample set,
         `sample_count` is the number of samples the sample set contains,
         `study_id` is the identifier for the study that generated the sample set,
         `study_url` is the URL of the study on the MalariaGEN website,
         `term_of_use_expiry` is the date when the terms of use expire,
         `terms_of_use_url` is the URL of the terms of use,
         `release` is the identifier of the release containing the sample set,
         `unrestricted_use_only` whether the sample set can be without restriction (e.g., if the terms of use have expired).
         If `unrestricted_use_only` was set to `True` then only sample sets with `unrestricted_use` set to `True` will be included.
         If `surveillance_use_only` was set to `True` then only sample sets that contain one or more samples with `is_surveillance` set to `True` will be included.
            """,
    )
    def sample_sets(
        self,
        release: Optional[base_params.release] = None,
    ) -> pd.DataFrame:
        return self._relevant_sample_sets(release=release)

    @check_types
    @doc(
        summary="Access a dataframe of available sample sets",
        returns="""A dataframe of available sample sets, one row per sample set. It contains five columns:
         `sample_set` is the name of the sample set,
         `sample_count` is the number of samples the sample set contains,
         `study_id` is the identifier for the study that generated the sample set,
         `study_url` is the URL of the study on the MalariaGEN website,
         `term_of_use_expiry` is the date when the terms of use expire,
         `terms_of_use_url` is the URL of the terms of use,
         `release` is the identifier of the release containing the sample set,
         `unrestricted_use_only` whether the sample set can be without restriction (e.g., if the terms of use have expired).
            """,
    )
    def _available_sample_sets(
        self,
        release: Optional[base_params.release] = None,
    ) -> pd.DataFrame:
        if release is None:
            # Retrieve sample sets from all available releases.
            release = self._available_releases

        if isinstance(release, str):
            # Retrieve sample sets for a single release.

            if release not in self._available_releases:
                raise ValueError(
                    f"Release is either not relevant or not available: {release!r}"
                )

            try:
                df = self._cache_available_sample_sets[release]

            except KeyError:
                # Read and cache dataframe for performance.
                df = self._read_sample_sets_manifest(single_release=release)
                self._cache_available_sample_sets[release] = df

        elif isinstance(release, Sequence):
            # Ensure no duplicates.
            releases = sorted(set(release))

            # Retrieve and concatenate sample sets from multiple releases.
            df = pd.concat(
                [self._available_sample_sets(release=r) for r in releases],
                axis=0,
                ignore_index=True,
            )

        else:
            raise TypeError

        # Return copy to ensure cached dataframes aren't modified by user.
        return df.copy()

    @check_types
    @doc(
        summary="Access a dataframe of relevant sample sets",
        returns="""A dataframe of relevant sample sets, one row per sample set. It contains five columns:
         `sample_set` is the name of the sample set,
         `sample_count` is the number of samples the sample set contains,
         `study_id` is the identifier for the study that generated the sample set,
         `study_url` is the URL of the study on the MalariaGEN website,
         `term_of_use_expiry` is the date when the terms of use expire,
         `terms_of_use_url` is the URL of the terms of use,
         `release` is the identifier of the release containing the sample set,
         `unrestricted_use_only` whether the sample set can be without restriction (e.g., if the terms of use have expired).
         If `unrestricted_use_only` was set to `True` then only sample sets with `unrestricted_use` set to `True` will be included.
         If `surveillance_use_only` was set to `True` then only sample sets that contain one or more samples with `is_surveillance` set to `True` will be included.
            """,
    )
    def _relevant_sample_sets(
        self,
        release: Optional[base_params.release] = None,
    ) -> pd.DataFrame:
        # Note: `release` must either be `None` or be one of `_relevant_releases`.
        # Otherwise this function will raise a `ValueError`.

        if release is None:
            # Retrieve sample sets from all relevant releases.
            release = self._relevant_releases

        if isinstance(release, str):
            # Retrieve sample sets for a single release.

            if release not in self._relevant_releases:
                raise ValueError(
                    f"Release is either not relevant or not available: {release!r}"
                )

            try:
                df = self._cache_sample_sets[release]

            except KeyError:
                # Read and cache dataframe for performance.
                df = self._read_sample_sets_manifest(single_release=release)

                # If unrestricted_use_only, restrict to sample sets with unrestricted_use.
                if "unrestricted_use" in df.columns and self._unrestricted_use_only:
                    df = df[df["unrestricted_use"].astype(bool)]

                # If surveillance_use_only, restrict to sample sets that contain one or more `is_surveillance` samples.
                if self._surveillance_use_only:
                    # Start a list of the relevant sample sets.
                    relevant_sample_sets = []

                    # For each of the DataFrame's sample sets...
                    release_sample_sets = df["sample_set"].to_list()
                    for sample_set in release_sample_sets:
                        # Determine whether this sample set has surveillance data.
                        if self._sample_set_has_surveillance_data(
                            sample_set=sample_set
                        ):
                            relevant_sample_sets.append(sample_set)

                    # Remove other sample sets from the DataFrame.
                    df = df[df["sample_set"].isin(relevant_sample_sets)]

                self._cache_sample_sets[release] = df

        elif isinstance(release, Sequence):
            # Ensure no duplicates.
            releases = sorted(set(release))

            # Retrieve and concatenate sample sets from multiple releases.
            df = pd.concat(
                [self._relevant_sample_sets(release=r) for r in releases],
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
        returns="The release the sample set is part of.",
    )
    def lookup_release(self, sample_set: base_params.sample_set) -> str:
        if self._cache_sample_set_to_release is None:
            df_sample_sets = self._available_sample_sets().set_index("sample_set")
            self._cache_sample_set_to_release = df_sample_sets["release"].to_dict()

        try:
            return self._cache_sample_set_to_release[sample_set]
        except KeyError:
            raise ValueError(
                f"No release found for sample set {sample_set!r}. This sample set might be unavailable or irrelevant with respect to settings."
            )

    @check_types
    @doc(
        summary="Find which study a sample set belongs to.",
        returns="The study the sample set belongs to.",
    )
    def lookup_study(self, sample_set: base_params.sample_set) -> str:
        if self._cache_sample_set_to_study is None:
            df_sample_sets = self._available_sample_sets().set_index("sample_set")
            self._cache_sample_set_to_study = df_sample_sets["study_id"].to_dict()
        try:
            return self._cache_sample_set_to_study[sample_set]
        except KeyError:
            raise ValueError(f"No study ID found for sample set {sample_set!r}")

    @check_types
    @doc(
        summary="Find the study info for a sample set.",
        returns="The info for the study the sample set belongs to.",
    )
    def lookup_study_info(self, sample_set: base_params.sample_set) -> dict:
        if self._cache_sample_set_to_study_info is None:
            df_sample_sets = self._available_sample_sets().set_index("sample_set")
            self._cache_sample_set_to_study_info = df_sample_sets[
                ["study_id", "study_url"]
            ].to_dict(orient="index")
        try:
            return self._cache_sample_set_to_study_info[sample_set]
        except KeyError:
            raise ValueError(f"No study info found for sample set {sample_set!r}")

    @check_types
    @doc(
        summary="Find the terms-of-use info for a sample set.",
        returns="The terms-of-use info for the sample set.",
    )
    def lookup_terms_of_use_info(self, sample_set: base_params.sample_set) -> dict:
        if self._cache_sample_set_to_terms_of_use_info is None:
            df_sample_sets = self._available_sample_sets().set_index("sample_set")
            self._cache_sample_set_to_terms_of_use_info = df_sample_sets[
                [
                    "terms_of_use_expiry_date",
                    "terms_of_use_url",
                    "unrestricted_use",
                ]
            ].to_dict(orient="index")
        try:
            return self._cache_sample_set_to_terms_of_use_info[sample_set]
        except KeyError:
            raise ValueError(
                f"No terms-of-use info found for sample set {sample_set!r}"
            )

    def _prep_sample_sets_param(
        self, *, sample_sets: Optional[base_params.sample_sets]
    ) -> List[str]:
        """Common handling for the `sample_sets` parameter. For convenience, we
        allow this to be a single sample set, or a list of sample sets, or a
        release identifier, or a list of release identifiers."""

        # Get the relevant sample sets as a list.
        all_relevant_sample_sets = self._relevant_sample_sets()["sample_set"].to_list()

        # If no sample sets are specified...
        if sample_sets is None:
            # Assume we want all relevant sample sets.
            prepared_sample_sets = all_relevant_sample_sets

        # Otherwise, if the sample sets are specified as a string...
        elif isinstance(sample_sets, str):
            # If the given string starts with the release major version number...
            if sample_sets.startswith(f"{self._major_version_number}."):
                # Assume the given string is a release.
                release = str(sample_sets)

                # Get the relevant sample sets for this release as a list.
                prepared_sample_sets = self._relevant_sample_sets(release=release)[
                    "sample_set"
                ].to_list()

            else:
                # Assume the given string is a single sample set identifier.
                # Put the single sample set identifier into a list, for consistency.
                prepared_sample_sets = [sample_sets]

        else:
            # Check that the given sample_sets is some kind of Sequence.
            # Otherwise, raise an error.
            if not isinstance(sample_sets, Sequence):
                sample_sets_type = type(sample_sets)
                raise ValueError(
                    f"Unsupported data type for sample_sets param: {sample_sets_type}"
                )

            # sample_sets is a kind of Sequence.
            seq = sample_sets

            # Start a list of prepared sample sets.
            prepared_sample_sets = []

            # For each item in the given Sequence...
            for seq_item in seq:
                # The item might be a release identifier.
                # Make a recursive call to reduce release identifiers into a list of sample sets.
                seq_item_sample_sets = self._prep_sample_sets_param(
                    sample_sets=seq_item
                )

                # Use `extend` rather than `append`, because we are adding a list to a list.
                prepared_sample_sets.extend(seq_item_sample_sets)

        # Remove duplicates from the list of sample sets and sort it.
        prepared_sample_sets = sorted(set(prepared_sample_sets))

        # Check for unavailable or irrelevant sample sets.
        if set(prepared_sample_sets) != set(all_relevant_sample_sets):
            for sample_set in prepared_sample_sets:
                if sample_set not in all_relevant_sample_sets:
                    raise ValueError(
                        f"Sample set {sample_set!r} not found. This sample set might be unavailable or irrelevant with respect to settings."
                    )

        return prepared_sample_sets

    def _prep_sample_query_param(
        self, *, sample_query: Optional[base_params.sample_query]
    ) -> Optional[base_params.sample_query]:
        """Common handling for the `sample_query` parameter."""

        # Return the same data type and default to the original value.
        prepped_sample_query: Optional[base_params.sample_query] = sample_query

        # If `_surveillance_use_only` then ensure there is an is_surveillance query criterion.
        if self._surveillance_use_only:
            is_surveillance_query_criterion = "is_surveillance == True"
            # If there is no query, then set it to the is_surveillance query criterion.
            if sample_query is None or sample_query.strip() == "":
                prepped_sample_query = is_surveillance_query_criterion
            else:
                # If the current query already ends with the is_surveillance query criterion, then keep it as it is.
                if sample_query.endswith(f" and {is_surveillance_query_criterion}"):
                    prepped_sample_query = sample_query
                else:
                    prepped_sample_query = (
                        f"{sample_query} and {is_surveillance_query_criterion}"
                    )

        return prepped_sample_query

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
