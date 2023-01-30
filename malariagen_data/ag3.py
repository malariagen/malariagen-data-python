import sys
import warnings
from bisect import bisect_left, bisect_right
from collections import Counter
from textwrap import dedent

import allel
import dask
import dask.array as da
import numba
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .anopheles import (
    DEFAULT_GENES_TRACK_HEIGHT,
    DEFAULT_GENOME_PLOT_SIZING_MODE,
    DEFAULT_GENOME_PLOT_WIDTH,
    AnophelesDataResource,
)

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

import malariagen_data  # used for .__version__

from .mjn import median_joining_network, mjn_graph
from .util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    CacheMiss,
    Region,
    da_from_zarr,
    init_zarr_store,
    locate_region,
    plotly_discrete_legend,
    xarray_concat,
)

# silence dask performance warnings
dask.config.set(**{"array.slicing.split_large_chunks": False})

MAJOR_VERSION_INT = 3
MAJOR_VERSION_GCS_STR = "v3"
CONFIG_PATH = "v3-config.json"

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
SITE_ANNOTATIONS_ZARR_PATH = (
    "reference/genome/agamp4/Anopheles-gambiae-PEST_SEQANNOTATION_AgamP4.12.zarr"
)
GENOME_REF_ID = "AgamP4"
GENOME_REF_NAME = "Anopheles gambiae (PEST)"

CONTIGS = "2R", "2L", "3R", "3L", "X"
DEFAULT_MAX_COVERAGE_VARIANCE = 0.2

PCA_RESULTS_CACHE_NAME = "ag3_pca_v1"
SNP_ALLELE_COUNTS_CACHE_NAME = "ag3_snp_allele_counts_v2"
FST_GWSS_CACHE_NAME = "ag3_fst_gwss_v1"
DEFAULT_SITE_MASK = "gamb_colu_arab"


class Ag3(AnophelesDataResource):
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
    _major_version_int = MAJOR_VERSION_INT
    _major_version_gcs_str = MAJOR_VERSION_GCS_STR
    _genome_fasta_path = GENOME_FASTA_PATH
    _genome_fai_path = GENOME_FAI_PATH
    _genome_zarr_path = GENOME_ZARR_PATH
    _genome_ref_id = GENOME_REF_ID
    _genome_ref_name = GENOME_REF_NAME
    _gcs_url = GCS_URL
    _pca_results_cache_name = PCA_RESULTS_CACHE_NAME
    _snp_allele_counts_results_cache_name = SNP_ALLELE_COUNTS_CACHE_NAME
    _fst_gwss_results_cache_name = FST_GWSS_CACHE_NAME
    _default_site_mask = DEFAULT_SITE_MASK
    _site_annotations_zarr_path = SITE_ANNOTATIONS_ZARR_PATH

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
        pre=False,
        **kwargs,  # used by simplecache, init_filesystem(url, **kwargs)
    ):

        super().__init__(
            url=url,
            config_path=CONFIG_PATH,
            cohorts_analysis=cohorts_analysis,
            site_filters_analysis=site_filters_analysis,
            bokeh_output_notebook=bokeh_output_notebook,
            results_cache=results_cache,
            log=log,
            debug=debug,
            show_progress=show_progress,
            check_location=check_location,
            pre=pre,
            **kwargs,  # used by simplecache, init_filesystem(url, **kwargs)
        )

        # set species analysis version - this is Ag specific currently, hence
        # not in parent class
        if species_analysis is None:
            self._species_analysis = self._config["DEFAULT_SPECIES_ANALYSIS"]
        else:
            self._species_analysis = species_analysis

        # set up caches
        self._cache_species_calls = dict()
        self._cache_cross_metadata = None
        self._cache_cnv_hmm = dict()
        self._cache_cnv_coverage_calls = dict()
        self._cache_cnv_discordant_read_calls = dict()
        self._cache_haplotypes = dict()
        self._cache_haplotype_sites = dict()
        self._cache_aim_variants = dict()

    @property
    def v3_wild(self):
        """Legacy, convenience property to access sample sets from the
        3.0 release, excluding the lab crosses."""
        return [
            x
            for x in self.sample_sets(release="3.0")["sample_set"].tolist()
            if x != "AG1000G-X"
        ]

    @staticmethod
    def _setup_taxon_colors(plot_kwargs=None):
        import plotly.express as px

        if plot_kwargs is None:
            plot_kwargs = dict()
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
        plot_kwargs.setdefault(
            "category_orders", {"taxon": list(taxon_color_map.keys())}
        )
        return plot_kwargs

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

    def _read_species_calls(self, *, sample_set):
        """Read species calls for a single sample set."""
        key = sample_set
        try:
            df = self._cache_species_calls[key]

        except KeyError:
            release = self._lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
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

    # TODO: generalise (species, cohorts) so we can abstract to parent class
    def _sample_metadata(self, *, sample_set):
        df = self._read_general_metadata(sample_set=sample_set)
        df_species = self._read_species_calls(sample_set=sample_set)
        df = df.merge(df_species, on="sample_id", sort=False)
        df_cohorts = self._read_cohort_metadata(sample_set=sample_set)
        df = df.merge(df_cohorts, on="sample_id", sort=False)
        return df

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

    def _site_mask_ids(self):
        if self._site_filters_analysis == "dt_20200416":
            return "gamb_colu_arab", "gamb_colu", "arab"
        else:
            raise ValueError

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
            release_path = self._release_to_path(release)
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
            release_path = self._release_to_path(release)
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
            release_path = self._release_to_path(release)
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
        self._check_param_min_cohort_size(min_cohort_size)
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
        title = f"Gene CNV frequencies ({self._region_str(region)})"
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
        coh_dict = self._locate_cohorts(cohorts=cohorts, df_samples=df_samples)

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
        df["label"] = self._pandas_apply(
            self._make_gene_cnv_label, df, columns=["gene_id", "gene_name", "cnv_type"]
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
            release_path = self._release_to_path(release)
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

        self._check_param_min_cohort_size(min_cohort_size)

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

        title = f"Gene CNV frequencies ({self._region_str(region)})"
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
        df_samples = self._prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        debug("group samples to make cohorts")
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        debug("build cohorts dataframe")
        df_cohorts = self._build_cohorts_from_sample_grouping(
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
        df_variants["label"] = self._pandas_apply(
            self._make_gene_cnv_label,
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
        self._add_frequency_ci(ds_out, ci_method)

        debug("tidy up display by sorting variables")
        ds_out = ds_out[sorted(ds_out)]

        return ds_out

    def plot_cnv_hmm_coverage_track(
        self,
        sample,
        region,
        sample_set=None,
        y_max="auto",
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
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
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            sizing_mode=sizing_mode,
            width=width,
            height=height,
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
        self._bokeh_style_genome_xaxis(fig, region.contig)
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
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
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
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            sizing_mode=sizing_mode,
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
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        debug("combine plots into a single figure")
        fig = bklay.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
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
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
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
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            sizing_mode=sizing_mode,
            width=width,
            height=plot_height,
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
        self._bokeh_style_genome_xaxis(fig, region.contig)
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
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
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
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            sizing_mode=sizing_mode,
            width=width,
            row_height=row_height,
            height=track_height,
            show=False,
        )
        fig1.xaxis.visible = False

        debug("plot genes track")
        fig2 = self.plot_genes(
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        debug("combine plots into a single figure")
        fig = bklay.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:
            bkplt.show(fig)

        return fig

    def _view_alignments_add_site_filters_tracks(
        self, *, contig, visibility_window, tracks
    ):
        debug = self._log.debug

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

    def _results_cache_add_analysis_params(self, params):
        super()._results_cache_add_analysis_params(params)
        # override parent class to add species analysis
        params["species_analysis"] = self._species_analysis

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
        release_path = self._release_to_path(release)
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
        calibration runs : list of numpy.ndarray
            A list of H12 calibration run arrays for each window size, containing
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
        fig = bkplt.figure(width=700, height=400, x_axis_type="log")
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
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
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
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            sizing_mode=sizing_mode,
            width=width,
            height=height,
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
        self._bokeh_style_genome_xaxis(fig, contig)

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
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
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
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            show=False,
        )

        fig1.xaxis.visible = False

        # plot genes
        fig2 = self.plot_genes(
            region=contig,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        # combine plots into a single figure
        fig = bklay.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        bkplt.show(fig)

    def h1x_gwss(
        self,
        contig,
        analysis,
        window_size,
        cohort1_query,
        cohort2_query,
        sample_sets=None,
        cohort_size=30,
        random_seed=42,
    ):
        """Run a H1X genome-wide scan to detect genome regions with
        shared selective sweeps between two cohorts.

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
        cohort1_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort2_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.

        Returns
        -------
        x : numpy.ndarray
            An array containing the window centre point genomic positions.
        h1x : numpy.ndarray
            An array with H1X statistic values for each window.

        """
        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = "ag3_h1x_gwss_v1"

        params = dict(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._h1x_gwss(**params)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        h1x = results["h1x"]

        return x, h1x

    def _h1x_gwss(
        self,
        contig,
        analysis,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        cohort_size,
        random_seed,
    ):

        # access haplotype datasets for each cohort
        ds1 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort1_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )
        ds2 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort2_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        # load data into memory
        gt1 = allel.GenotypeDaskArray(ds1["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes for cohort 1"):
            ht1 = gt1.to_haplotypes().compute()
        gt2 = allel.GenotypeDaskArray(ds2["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes for cohort 2"):
            ht2 = gt2.to_haplotypes().compute()
        pos = ds1["variant_position"].values

        # run H1X scan
        h1x = _moving_h1x(ht1, ht2, size=window_size)

        # compute window midpoints
        x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, h1x=h1x)

        return results

    def plot_h1x_gwss_track(
        self,
        contig,
        analysis,
        window_size,
        cohort1_query,
        cohort2_query,
        sample_sets=None,
        cohort_size=30,
        random_seed=42,
        title=None,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        height=200,
        show=True,
        x_range=None,
    ):
        """Run and plot a H1X genome-wide scan to detect genome regions
        with shared selective sweeps between two cohorts.

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
        cohort1_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort2_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
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
            A plot showing windowed H1X statistic across chosen contig.

        """

        import bokeh.models as bkmod
        import bokeh.plotting as bkplt

        # compute H1X
        x, h1x = self.h1x_gwss(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort_size=cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
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
            title = f"Cohort 1: {cohort1_query}\nCohort 2: {cohort2_query}"
        fig = bkplt.figure(
            title=title,
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, 1),
        )

        # plot H1X
        fig.circle(
            x=x,
            y=h1x,
            size=3,
            line_width=0.5,
            line_color="black",
            fill_color=None,
        )

        # tidy up the plot
        fig.yaxis.axis_label = "H1X"
        fig.yaxis.ticker = [0, 1]
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:
            bkplt.show(fig)

        return fig

    def plot_h1x_gwss(
        self,
        contig,
        analysis,
        window_size,
        cohort1_query,
        cohort2_query,
        sample_sets=None,
        cohort_size=30,
        random_seed=42,
        title=None,
        sizing_mode=DEFAULT_GENOME_PLOT_SIZING_MODE,
        width=DEFAULT_GENOME_PLOT_WIDTH,
        track_height=190,
        genes_height=DEFAULT_GENES_TRACK_HEIGHT,
    ):
        """Run and plot a H1X genome-wide scan to detect genome regions
        with shared selective sweeps between two cohorts.

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
        cohort1_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        cohort2_query : str
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        title : str, optional
            If provided, title string is used to label plot.
        sizing_mode : str, optional
            Bokeh plot sizing mode, see https://docs.bokeh.org/en/latest/docs/user_guide/layout.html#sizing-modes
        width : int, optional
            Plot width in pixels (px).
        track_height : int. optional
            GWSS track height in pixels (px).
        genes_height : int. optional
            Gene track height in pixels (px).

        Returns
        -------
        fig : figure
            A plot showing windowed H1X statistic with gene track on x-axis.

        """

        import bokeh.layouts as bklay
        import bokeh.plotting as bkplt

        # gwss track
        fig1 = self.plot_h1x_gwss_track(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            show=False,
        )

        fig1.xaxis.visible = False

        # plot genes
        fig2 = self.plot_genes(
            region=contig,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        # combine plots into a single figure
        fig = bklay.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        bkplt.show(fig)

    def plot_haplotype_clustering(
        self,
        region,
        analysis,
        sample_sets=None,
        sample_query=None,
        color=None,
        symbol=None,
        linkage_method="single",
        count_sort=True,
        distance_sort=False,
        cohort_size=None,
        random_seed=42,
        width=1000,
        height=500,
        **kwargs,
    ):
        """Hierarchically cluster haplotypes in region and produce an interactive plot.

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
        color : str, optional
            Identifies a column in the sample metadata which determines the colour
            of dendrogram leaves (haplotypes).
        symbol : str, optional
            Identifies a column in the sample metadata which determines the shape
            of dendrogram leaves (haplotypes).
        linkage_method: str, optional
            The linkage algorithm to use, valid options are 'single', 'complete',
            'average', 'weighted', 'centroid', 'median' and 'ward'. See the Linkage
            Methods section of the scipy.cluster.hierarchy.linkage docs for full
            descriptions.
        count_sort: bool, optional
            For each node n, the order (visually, from left-to-right) n's two descendant
            links are plotted is determined by this parameter. If True, the child with
            the minimum number of original objects in its cluster is plotted first. Note
            distance_sort and count_sort cannot both be True.
        distance_sort: bool, optional
            For each node n, the order (visually, from left-to-right) n's two descendant
            links are plotted is determined by this parameter. If True, The child with the
            minimum distance between its direct descendants is plotted first.
        cohort_size : int, optional
            If provided, randomly down-sample to the given cohort size.
        random_seed : int, optional
            Random seed used for down-sampling.
        width : int, optional
            The figure width in pixels
        height: int, optional
            The figure height in pixels

        Returns
        -------
        fig : Figure
            Plotly figure.

        """
        import plotly.express as px
        from scipy.cluster.hierarchy import linkage

        from .plotly_dendrogram import create_dendrogram

        debug = self._log.debug

        ds_haps = self.haplotypes(
            region=region,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )
        debug("align sample metadata with haplotypes")
        phased_samples = ds_haps["sample_id"].values.tolist()
        df_samples_phased = (
            df_samples.set_index("sample_id").loc[phased_samples].reset_index()
        )

        debug("set up plotting options")
        hover_data = [
            "sample_id",
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
            template="simple_white",
            hover_name="sample_id",
            hover_data=hover_data,
            render_mode="svg",
        )

        debug("special handling for taxon color")
        if color == "taxon":
            self._setup_taxon_colors(plot_kwargs)

        debug("apply any user overrides")
        plot_kwargs.update(kwargs)

        debug("Create dendrogram with plotly")
        # set labels as the index which we extract to reorder metadata
        leaf_labels = np.arange(ht.shape[1])
        # get the max distance, required to set xmin, xmax, which we need xmin to be slightly below 0
        max_dist = _get_max_hamming_distance(
            ht.T, metric="hamming", linkage_method=linkage_method
        )
        # noinspection PyTypeChecker
        fig = create_dendrogram(
            ht.T,
            distfun=lambda x: _hamming_to_snps(x),
            linkagefun=lambda x: linkage(x, method=linkage_method),
            # FIXME: expected type 'list', got 'ndarray'
            labels=leaf_labels,
            color_threshold=0,
            count_sort=count_sort,
            distance_sort=distance_sort,
        )
        fig.update_traces(
            hoverinfo="y",
            line=dict(width=0.5, color="black"),
        )

        title_lines = []
        if sample_sets is not None:
            title_lines.append(f"sample sets: {sample_sets}")
        if sample_query is not None:
            title_lines.append(f"sample query: {sample_query}")
        title_lines.append(f"genomic region: {region} ({ht.shape[0]} SNPs)")
        title = "<br>".join(title_lines)

        fig.update_layout(
            width=width,
            height=height,
            title=title,
            autosize=True,
            hovermode="closest",
            plot_bgcolor="white",
            yaxis_title="Distance (no. SNPs)",
            xaxis_title="Haplotypes",
            showlegend=True,
        )

        # Repeat the dataframe so there is one row of metadata for each haplotype
        df_samples_phased_haps = pd.DataFrame(
            np.repeat(df_samples_phased.values, 2, axis=0)
        )
        df_samples_phased_haps.columns = df_samples_phased.columns
        # select only columns in hover_data
        df_samples_phased_haps = df_samples_phased_haps[hover_data]
        debug("Reorder haplotype metadata to align with haplotype clustering")
        df_samples_phased_haps = df_samples_phased_haps.loc[
            fig.layout.xaxis["ticktext"]
        ]
        fig.update_xaxes(mirror=False, showgrid=True, showticklabels=False, ticks="")
        fig.update_yaxes(
            mirror=False, showgrid=True, showline=True, range=[-2, max_dist + 1]
        )

        debug("Add scatter plot with hover text")
        fig.add_traces(
            list(
                px.scatter(
                    df_samples_phased_haps,
                    x=fig.layout.xaxis["tickvals"],
                    y=np.repeat(-1, len(ht.T)),
                    color=color,
                    symbol=symbol,
                    **plot_kwargs,
                ).select_traces()
            )
        )

        return fig

    def plot_haplotype_network(
        self,
        region,
        analysis,
        sample_sets=None,
        sample_query=None,
        max_dist=2,
        color=None,
        color_discrete_sequence=None,
        color_discrete_map=None,
        category_orders=None,
        node_size_factor=50,
        server_mode="inline",
        height=650,
        width="100%",
        layout="cose",
        layout_params=None,
        server_port=None,
    ):
        """Construct a median-joining haplotype network and display it using
        Cytoscape.

        A haplotype network provides a visualisation of the genetic distance
        between haplotypes. Each node in the network represents a unique
        haplotype. The size (area) of the node is scaled by the number of
        times that unique haplotype was observed within the selected samples.
        A connection between two nodes represents a single SNP difference
        between the corresponding haplotypes.

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
        max_dist : int, optional
            Join network components up to a maximum distance of 2 SNP
            differences.
        color : str, optional
            Identifies a column in the sample metadata which determines the colour
            of pie chart segments within nodes.
        color_discrete_sequence : list, optional
            Provide a list of colours to use.
        color_discrete_map : dict, optional
            Provide an explicit mapping from values to colours.
        category_orders : list, optional
            Control the order in which values appear in the legend.
        node_size_factor : int, optional
            Control the sizing of nodes.
        server_mode : {"inline", "external", "jupyterlab"}
            Controls how the Jupyter Dash app will be launched. See
            https://medium.com/plotly/introducing-jupyterdash-811f1f57c02e for
            more information.
        height : int, optional
            Height of the plot.
        width : int, optional
            Width of the plot.
        layout : str
            Name of the network layout to use to position nodes.
        layout_params
            Additional parameters to the layout algorithm.
        server_port
            Manually override the port on which the Dash app will run.

        Returns
        -------
        app
            The running Dash app.

        """

        from itertools import cycle

        # FIXME: unresolved references
        import dash_cytoscape as cyto
        import plotly.express as px
        from dash import dcc, html
        from dash.dependencies import Input, Output
        from jupyter_dash import JupyterDash

        if layout != "cose":
            cyto.load_extra_layouts()
        # leave this for user code, if needed (doesn't seem necessary on colab)
        # JupyterDash.infer_jupyter_proxy_config()

        debug = self._log.debug

        debug("access haplotypes dataset")
        ds_haps = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            analysis=analysis,
        )

        debug("access sample metadata")
        df_samples = self.sample_metadata(
            sample_query=sample_query, sample_sets=sample_sets
        )

        debug("setup haplotype metadata")
        samples_phased = ds_haps["sample_id"].values
        df_samples_phased = (
            df_samples.set_index("sample_id").loc[samples_phased].reset_index()
        )
        df_haps = df_samples_phased.loc[df_samples_phased.index.repeat(2)].reset_index(
            drop=True
        )

        debug("load haplotypes")
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        debug("count alleles and select segregating sites")
        ac = gt.count_alleles(max_allele=1)
        loc_seg = ac.is_segregating()
        ht_seg = ht[loc_seg]

        debug("identify distinct haplotypes")
        ht_distinct_sets = ht_seg.distinct()
        # find indices of distinct haplotypes - just need one per set
        ht_distinct_indices = [min(s) for s in ht_distinct_sets]
        # reorder by index - TODO is this necessary?
        ix = np.argsort(ht_distinct_indices)
        ht_distinct_indices = [ht_distinct_indices[i] for i in ix]
        ht_distinct_sets = [ht_distinct_sets[i] for i in ix]
        # obtain an array of distinct haplotypes
        ht_distinct = ht_seg.take(ht_distinct_indices, axis=1)
        # count how many observations per distinct haplotype
        ht_counts = [len(s) for s in ht_distinct_sets]

        debug("construct median joining network")
        ht_distinct_mjn, edges, alt_edges = median_joining_network(
            ht_distinct, max_dist=max_dist
        )
        edges = np.triu(edges)
        alt_edges = np.triu(alt_edges)

        debug("setup colors")
        color_values = None
        color_values_display = None
        color_discrete_map_display = None
        ht_color_counts = None
        if color is not None:

            # sanitise color column - necessary to avoid grey pie chart segments
            df_haps["partition"] = df_haps[color].str.replace(r"\W", "", regex=True)

            # extract all unique values of the color column
            color_values = df_haps["partition"].unique()
            color_values_mapping = dict(zip(df_haps["partition"], df_haps[color]))
            color_values_display = [color_values_mapping[c] for c in color_values]

            # count color values for each distinct haplotype
            ht_color_counts = [
                df_haps.iloc[list(s)]["partition"].value_counts().to_dict()
                for s in ht_distinct_sets
            ]

            if color == "taxon":
                # special case, standardise taxon colors and order
                color_params = self._setup_taxon_colors()
                color_discrete_map = color_params["color_discrete_map"]
                color_discrete_map_display = color_discrete_map
                category_orders = color_params["category_orders"]

            elif color_discrete_map is None:

                # set up a color palette
                if color_discrete_sequence is None:
                    if len(color_values) <= 10:
                        color_discrete_sequence = px.colors.qualitative.Plotly
                    else:
                        color_discrete_sequence = px.colors.qualitative.Alphabet

                # map values to colors
                color_discrete_map = {
                    v: c for v, c in zip(color_values, cycle(color_discrete_sequence))
                }
                color_discrete_map_display = {
                    v: c
                    for v, c in zip(
                        color_values_display, cycle(color_discrete_sequence)
                    )
                }

        debug("construct graph")
        anon_width = np.sqrt(0.3 * node_size_factor)
        graph_nodes, graph_edges = mjn_graph(
            ht_distinct=ht_distinct,
            ht_distinct_mjn=ht_distinct_mjn,
            ht_counts=ht_counts,
            ht_color_counts=ht_color_counts,
            color=color,
            color_values=color_values,
            edges=edges,
            alt_edges=alt_edges,
            node_size_factor=node_size_factor,
            anon_width=anon_width,
        )

        debug("prepare graph data for cytoscape")
        elements = [{"data": n} for n in graph_nodes] + [
            {"data": e} for e in graph_edges
        ]

        debug("define node style")
        node_stylesheet = {
            "selector": "node",
            "style": {
                "width": "data(width)",
                "height": "data(width)",
                "pie-size": "100%",
            },
        }
        if color:
            # here are the styles which control the display of nodes as pie
            # charts
            for i, (v, c) in enumerate(color_discrete_map.items()):
                node_stylesheet["style"][f"pie-{i + 1}-background-color"] = c
                node_stylesheet["style"][
                    f"pie-{i + 1}-background-size"
                ] = f"mapData({v}, 0, 100, 0, 100)"
        debug(node_stylesheet)

        debug("define edge style")
        edge_stylesheet = {
            "selector": "edge",
            "style": {"curve-style": "bezier", "width": 2, "opacity": 0.5},
        }

        debug("define style for selected node")
        selected_stylesheet = {
            "selector": ":selected",
            "style": {
                "border-width": "3px",
                "border-style": "solid",
                "border-color": "black",
            },
        }

        debug("create figure legend")
        if color is not None:
            legend_fig = plotly_discrete_legend(
                color=color,
                color_values=color_values_display,
                color_discrete_map=color_discrete_map_display,
                category_orders=category_orders,
            )
            legend_component = dcc.Graph(
                id="legend",
                figure=legend_fig,
                config=dict(
                    displayModeBar=False,
                ),
            )
        else:
            legend_component = html.Div()

        debug("define cytoscape component")
        if layout_params is None:
            graph_layout_params = dict()
        else:
            graph_layout_params = layout_params.copy()
        graph_layout_params["name"] = layout
        # FIXME: expected type 'str', got 'int'
        # noinspection PyTypeChecker
        graph_layout_params.setdefault("padding", 10)
        # FIXME: expected type 'str', got 'bool'
        # noinspection PyTypeChecker
        graph_layout_params.setdefault("animate", False)

        cytoscape_component = cyto.Cytoscape(
            id="cytoscape",
            elements=elements,
            layout=graph_layout_params,
            stylesheet=[
                node_stylesheet,
                edge_stylesheet,
                selected_stylesheet,
            ],
            style={
                # width and height needed to get cytoscape component to display
                "width": "100%",
                "height": "100%",
                "background-color": "white",
            },
            # enable selecting multiple nodes with shift click and drag
            boxSelectionEnabled=True,
            # prevent accidentally zooming out to oblivion
            minZoom=0.1,
        )

        debug("create dash app")
        app = JupyterDash(
            "dash-cytoscape-network",
            # this stylesheet is used to provide support for a rows and columns
            # layout of the components
            external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
        )
        # this is an optimisation, it's generally faster to serve script files from CDN
        app.scripts.config.serve_locally = False
        app.layout = html.Div(
            [
                html.Div(
                    cytoscape_component,
                    className="nine columns",
                    style={
                        # required to get cytoscape component to show ...
                        # multiply by factor <1 to prevent scroll overflow
                        "height": f"{height * .93}px",
                        "border": "1px solid black",
                    },
                ),
                html.Div(
                    legend_component,
                    className="three columns",
                    style={
                        "height": f"{height * .93}px",
                    },
                ),
                html.Div(id="output"),
            ],
        )

        debug(
            "define a callback function to display information about the selected node"
        )

        @app.callback(Output("output", "children"), Input("cytoscape", "tapNodeData"))
        def display_tap_node_data(data):
            if data is None:
                return "Click or tap a node for more information."
            else:
                n = data["count"]
                text = f"No. haplotypes: {n}"
                selected_color_data = {
                    color_v_display: int(data.get(color_v, 0) * n / 100)
                    for color_v, color_v_display in zip(
                        color_values, color_values_display
                    )
                }
                selected_color_data = sorted(
                    selected_color_data.items(), key=lambda item: item[1], reverse=True
                )
                color_texts = [
                    f"{color_v}: {color_n}"
                    for color_v, color_n in selected_color_data
                    if color_n > 0
                ]
                if color_texts:
                    color_texts = "; ".join(color_texts)
                    text += f" ({color_texts})"
                return text

        debug("launch the dash app")
        run_params = dict()
        if server_mode is not None:
            run_params["mode"] = server_mode
        if server_port is not None:
            run_params["port"] = server_port
        if height is not None:
            run_params["height"] = height
        if width is not None:
            run_params["width"] = width
        return app.run_server(**run_params)


def _hamming_to_snps(h):
    """
    Cluster haplotype array and return the number of SNP differences
    """
    from scipy.spatial.distance import pdist

    dist = pdist(h, metric="hamming")
    dist *= h.shape[1]
    return dist


def _get_max_hamming_distance(h, metric="hamming", linkage_method="single"):
    """
    Find the maximum hamming distance between haplotypes
    """
    from scipy.cluster.hierarchy import linkage

    # FIXME: variable in function should be lowercase
    Z = linkage(h, metric=metric, method=linkage_method)

    # Get the distances column
    dists = Z[:, 2]
    # Convert to the number of SNP differences
    dists *= h.shape[1]
    # Return the maximum
    return dists.max()


def _haplotype_frequencies(h):
    """Compute haplotype frequencies, returning a dictionary that maps
    haplotype hash values to frequencies."""
    n = h.shape[1]
    hashes = [hash(h[:, i].tobytes()) for i in range(n)]
    counts = Counter(hashes)
    freqs = {key: count / n for key, count in counts.items()}
    return freqs


def _haplotype_joint_frequencies(ha, hb):
    """Compute the joint frequency of haplotypes in two difference
    cohorts. Returns a dictionary mapping haplotype hash values to
    the product of frequencies in each cohort."""
    frqa = _haplotype_frequencies(ha)
    frqb = _haplotype_frequencies(hb)
    keys = set(frqa.keys()) | set(frqb.keys())
    joint_freqs = {key: frqa.get(key, 0) * frqb.get(key, 0) for key in keys}
    return joint_freqs


def _h1x(ha, hb):
    """Compute H1X, the sum of joint haplotype frequencies between
    two cohorts, which is a summary statistic useful for detecting
    shared selective sweeps."""
    jf = _haplotype_joint_frequencies(ha, hb)
    return np.sum(list(jf.values()))


def _moving_h1x(ha, hb, size, start=0, stop=None, step=None):
    """Compute H1X in moving windows.
    Parameters
    ----------
    ha : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the first cohort.
    hb : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the second cohort.
    size : int
        The window size (number of variants).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    step : int, optional
        The number of variants between start positions of windows. If not
        given, defaults to the window size, i.e., non-overlapping windows.
    Returns
    -------
    h1x : ndarray, float, shape (n_windows,)
        H1X values (sum of squares of joint haplotype frequencies).
    """

    assert ha.ndim == hb.ndim == 2
    assert ha.shape[0] == hb.shape[0]

    # construct moving windows
    windows = allel.index_windows(ha, size, start, stop, step)

    # compute statistics for each window
    out = np.array([_h1x(ha[i:j], hb[i:j]) for i, j in windows])

    return out


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
