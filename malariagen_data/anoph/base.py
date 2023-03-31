import json
from collections.abc import Mapping
from typing import IO, Dict, Optional, Sequence, Tuple, Union

import bokeh.io
import pandas as pd
from numpydoc_decorator import doc
from typing_extensions import Annotated, TypeAlias

from ..util import LoggingHelper, Region, check_colab_location, init_filesystem


class base_params:
    """Parameter definitions common to many functions."""

    contig: TypeAlias = Annotated[
        str,
        """
        Reference genome contig name. See the `contigs` property for valid contig
        names.
        """,
    ]
    region: TypeAlias = Annotated[
        Union[str, Region],
        """
        Region of the reference genome. Can be a contig name, region string
        (formatted like "{contig}:{start}-{end}"), or identifier of a genome
        feature such as a gene or transcript.
        """,
    ]
    release: TypeAlias = Annotated[
        Union[str, Sequence[str]],
        "Release version identifier.",
    ]
    sample_set: TypeAlias = Annotated[
        str,
        "Sample set identifier.",
    ]
    sample_sets: TypeAlias = Annotated[
        Union[Sequence[str], str],
        """
        List of sample sets and/or releases. Can also be a single sample set or
        release.
        """,
    ]
    sample_query: TypeAlias = Annotated[
        str,
        """
        A pandas query string to be evaluated against the sample metadata.
        """,
    ]
    cohort1_query: TypeAlias = Annotated[
        str,
        """
        A pandas query string to be evaluated against the sample metadata,
        to select samples for the first cohort.
        """,
    ]
    cohort2_query: TypeAlias = Annotated[
        str,
        """
        A pandas query string to be evaluated against the sample metadata,
        to select samples for the second cohort.
        """,
    ]
    site_mask: TypeAlias = Annotated[
        str,
        """
        Which site filters mask to apply. See the `site_mask_ids` property for
        available values.
        """,
    ]
    site_class: TypeAlias = Annotated[
        str,
        """
        Select sites belonging to one of the following classes: CDS_DEG_4,
        (4-fold degenerate coding sites), CDS_DEG_2_SIMPLE (2-fold simple
        degenerate coding sites), CDS_DEG_0 (non-degenerate coding sites),
        INTRON_SHORT (introns shorter than 100 bp), INTRON_LONG (introns
        longer than 200 bp), INTRON_SPLICE_5PRIME (intron within 2 bp of
        5' splice site), INTRON_SPLICE_3PRIME (intron within 2 bp of 3'
        splice site), UTR_5PRIME (5' untranslated region), UTR_3PRIME (3'
        untranslated region), INTERGENIC (intergenic, more than 10 kbp from
        a gene).
        """,
    ]
    cohort_size: TypeAlias = Annotated[
        int,
        """
        Randomly down-sample to this value if the number of samples in the
        cohort is greater. Raise an error if the number of samples is less
        than this value.
        """,
    ]
    min_cohort_size: TypeAlias = Annotated[
        int,
        """
        Minimum cohort size. Raise an error if the number of samples is
        less than this value.
        """,
    ]
    max_cohort_size: TypeAlias = Annotated[
        int,
        """
        Randomly down-sample to this value if the number of samples in the
        cohort is greater.
        """,
    ]
    random_seed: TypeAlias = Annotated[
        int,
        "Random seed used for reproducible down-sampling.",
    ]
    transcript: TypeAlias = Annotated[
        str,
        "Gene transcript identifier.",
    ]
    cohort: TypeAlias = Annotated[
        Union[str, Tuple[str, str]],
        """
        Either a string giving one of the predefined cohort labels, or a
        pair of strings giving a custom cohort label and a sample query.
        """,
    ]
    cohorts: TypeAlias = Annotated[
        Union[str, Mapping[str, str]],
        """
        Either a string giving the name of a predefined cohort set (e.g.,
        "admin1_month") or a dict mapping custom cohort labels to sample
        queries.
        """,
    ]
    n_jack: TypeAlias = Annotated[
        int,
        """
        Number of blocks to divide the data into for the block jackknife
        estimation of confidence intervals. N.B., larger is not necessarily
        better.
        """,
    ]
    confidence_level: TypeAlias = Annotated[
        float,
        """
        Confidence level to use for confidence interval calculation. E.g., 0.95
        means 95% confidence interval.
        """,
    ]
    field: TypeAlias = Annotated[str, "Name of array or column to access."]
    inline_array: TypeAlias = Annotated[
        bool,
        "Passed through to dask `from_array()`.",
    ]
    inline_array_default: inline_array = True
    chunks: TypeAlias = Annotated[
        str,
        """
        If 'auto' let dask decide chunk size. If 'native' use native zarr
        chunks. Also, can be a target size, e.g., '200 MiB'.
        """,
    ]
    chunks_default: chunks = "native"


class AnophelesBase:
    def __init__(
        self,
        *,
        url: str,
        config_path: str,
        bokeh_output_notebook: bool,
        log: Optional[Union[str, IO]],
        debug: bool,
        show_progress: bool,
        check_location: bool,
        pre: bool,
        gcs_url: str,
        major_version_number: int,
        major_version_path: str,
        **storage_kwargs,
    ):
        self._url = url
        self._config_path = config_path
        self._debug = debug
        self._show_progress = show_progress
        self._pre = pre
        self._gcs_url = gcs_url
        self._major_version_number = major_version_number
        self._major_version_path = major_version_path

        # Set up logging.
        self._log = LoggingHelper(name=__name__, out=log, debug=debug)

        # Set up fsspec filesystem.
        self._fs, self._base_path = init_filesystem(url, **storage_kwargs)

        # Lazily load config.
        self._config: Optional[Dict] = None

        # Get bokeh to output plots to the notebook - this is a common gotcha,
        # users forget to do this and wonder why bokeh plots don't show.
        if bokeh_output_notebook:
            bokeh.io.output_notebook(hide_banner=True)

        # Check colab location is in the US.
        if check_location:
            self._client_details = check_colab_location(
                gcs_url=self._gcs_url, url=self._url
            )
        else:
            self._client_details = None

        # Set up cache attributes.
        self._cache_releases: Optional[Tuple[str, ...]] = None
        self._cache_sample_sets: Dict[str, pd.DataFrame] = dict()
        self._cache_sample_set_to_release: Optional[Dict[str, str]] = None

    def open_file(self, path):
        full_path = f"{self._base_path}/{path}"
        return self._fs.open(full_path)

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
        sub_dirs = [p.split("/")[-1] for p in self._fs.ls(self._base_path)]
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
            releases = list(set(release))

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
