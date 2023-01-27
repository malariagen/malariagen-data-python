import sys

from .anopheles import AnophelesDataResource

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

import malariagen_data  # used for .__version__

MAJOR_VERSION_INT = 1
MAJOR_VERSION_GCS_STR = "v1.0"
CONFIG_PATH = "v1.0-config.json"

GCS_URL = "gs://vo_afun_release/"

GENOME_FASTA_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa"
)
GENOME_FAI_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa.fai"
)
GENOME_ZARR_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.zarr"
)
SITE_ANNOTATIONS_ZARR_PATH = "reference/genome/idAnoFuneDA-416_04/Anopheles-funestus-DA-416_04_1_SEQANNOTATION.zarr"
GENOME_REF_ID = "idAnoFuneDA-416_04"
GENOME_REF_NAME = "Anopheles funestus"

CONTIGS = "2RL", "3RL", "X"

PCA_RESULTS_CACHE_NAME = "af1_pca_v1"
SNP_ALLELE_COUNTS_CACHE_NAME = "af1_snp_allele_counts_v2"
FST_GWSS_CACHE_NAME = "af1_fst_gwss_v1"
DEFAULT_SITE_MASK = "funestus"


class Af1(AnophelesDataResource):
    """Provides access to data from Af1.x releases.

    Parameters
    ----------
    url : str
        Base path to data. Give "gs://vo_afun_release/" to use Google Cloud
        Storage, or a local path on your file system if data have been
        downloaded.
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
        >>> af1 = malariagen_data.Af1()

    Access data downloaded to a local file system:

        >>> af1 = malariagen_data.Af1("/local/path/to/vo_afun_release/")

    Access data from Google Cloud Storage, with caching on the local file system
    in a directory named "gcs_cache":

        >>> af1 = malariagen_data.Af1(
        ...     "simplecache::gs://vo_afun_release",
        ...     simplecache=dict(cache_storage="gcs_cache"),
        ... )

    Set up caching of some longer-running computations on the local file system,
    in a directory named "results_cache":

        >>> af1 = malariagen_data.Af1(results_cache="results_cache")

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
    _cohorts_analysis = None
    _site_filters_analysis = None

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

    @staticmethod
    def _setup_taxon_colors(plot_kwargs=None):
        import plotly.express as px

        if plot_kwargs is None:
            plot_kwargs = dict()
        taxon_palette = px.colors.qualitative.Plotly
        taxon_color_map = {
            "funestus": taxon_palette[0],
        }
        plot_kwargs.setdefault("color_discrete_map", taxon_color_map)
        plot_kwargs.setdefault(
            "category_orders", {"taxon": list(taxon_color_map.keys())}
        )
        return plot_kwargs

    def __repr__(self):
        text = (
            f"<MalariaGEN Af1 API client>\n"
            f"Storage URL             : {self._url}\n"
            f"Data releases available : {', '.join(self.releases)}\n"
            f"Results cache           : {self._results_cache}\n"
            f"Cohorts analysis        : {self._cohorts_analysis}\n"
            f"Site filters analysis   : {self._site_filters_analysis}\n"
            f"Software version        : malariagen_data {malariagen_data.__version__}\n"
            f"Client location         : {self._client_location}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net."
        )
        # TODO: API doc https://malariagen.github.io/vector-data/af1/api.html
        return text

    def _repr_html_(self):
        # TODO: See also the <a href="https://malariagen.github.io/vector-data/af1/api.html">Af1 API docs</a>.
        html = f"""
            <table class="malariagen-af1">
                <thead>
                    <tr>
                        <th style="text-align: left" colspan="2">MalariaGEN Af1 API client</th>
                    </tr>
                    <tr><td colspan="2" style="text-align: left">
                        Please note that data are subject to terms of use,
                        for more information see <a href="https://www.malariagen.net/data">
                        the MalariaGEN website</a> or contact data@malariagen.net.
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

    def _sample_metadata(self, *, sample_set):
        df = self._read_general_metadata(sample_set=sample_set)
        df_cohorts = self._read_cohort_metadata(sample_set=sample_set)
        df = df.merge(df_cohorts, on="sample_id", sort=False)
        return df

    def _transcript_to_gene_name(self, transcript):
        df_genome_features = self.genome_features().set_index("ID")
        rec_transcript = df_genome_features.loc[transcript]
        parent = rec_transcript["Parent"]

        # E.g. manual overrides (used in Ag3)
        # if parent == "AGAP004707":
        #     parent_name = "Vgsc/para"
        # else:
        #     parent_name = rec_parent["Name"]

        # Note: Af1 doesn't have the "Name" attribute
        # rec_parent = df_genome_features.loc[parent]
        # parent_name = rec_parent["Name"]
        parent_name = parent

        return parent_name

    def _site_mask_ids(self):
        if self._site_filters_analysis == "dt_20200416":
            return ["funestus"]
        elif self._site_filters_analysis == "sc_20220908":
            return ["funestus"]
        else:
            raise ValueError

    def genome_features(
        self, region=None, attributes=("ID", "Parent", "Note", "description")
    ):
        """Access genome feature annotations.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Contig name (e.g., "2RL"), gene name (e.g., "LOC125767311"), genomic
            region defined with coordinates (e.g., "2RL:44,989,425-44,998,059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["2RL", "3RL"].
        attributes : list of str, optional
            Attribute keys to unpack into columns. Provide "*" to unpack all
            attributes.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of genome annotations, one row per feature.

        """

        # Here we override the superclass implementation in order to provide a
        # different default value for the `attributes` parameter, because the
        # genome annotations don't include a "Name" attribute but do include a
        # "Note" attribute which is probably useful to include instead.
        #
        # Also, we take the opportunity to customise the docstring to use
        # examples specific to funestus.
        #
        # See also https://github.com/malariagen/malariagen-data-python/issues/306

        return super().genome_features(region=region, attributes=attributes)

    def _plot_genes_setup_data(self, *, region):

        # Here we override the superclass implementation because the
        # gene annotations don't include a "Name" attribute.
        #
        # Also, the type needed is "protein_coding_gene".

        df_genome_features = self.genome_features(
            region=region, attributes=["ID", "Parent", "description"]
        )
        data = df_genome_features.query("type == 'protein_coding_gene'").copy()

        tooltips = [
            ("ID", "@ID"),
            ("Description", "@description"),
            ("Location", "@contig:@start{,}-@end{,}"),
        ]

        return data, tooltips

    def _view_alignments_add_site_filters_tracks(
        self, *, contig, visibility_window, tracks
    ):
        # Do nothing for now, because we don't have VCFs for the site filters
        # https://github.com/malariagen/vobs-funestus/issues/251
        # TODO implement when VCFs are available
        pass
