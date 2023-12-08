import sys

import malariagen_data  # used for .__version__

from .anopheles import AnophelesDataResource

MAJOR_VERSION_NUMBER = 1
MAJOR_VERSION_PATH = "v1.0"
CONFIG_PATH = "v1.0-config.json"
GCS_URL = "gs://vo_afun_release/"
PCA_RESULTS_CACHE_NAME = "af1_pca_v1"
FST_GWSS_CACHE_NAME = "af1_fst_gwss_v1"
H12_CALIBRATION_CACHE_NAME = "af1_h12_calibration_v1"
H12_GWSS_CACHE_NAME = "af1_h12_gwss_v1"
G123_GWSS_CACHE_NAME = "af1_g123_gwss_v1"
XPEHH_GWSS_CACHE_NAME = "af1_xpehh_gwss_v1"
G123_CALIBRATION_CACHE_NAME = "af1_g123_calibration_v1"
H1X_GWSS_CACHE_NAME = "af1_h1x_gwss_v1"
IHS_GWSS_CACHE_NAME = "af1_ihs_gwss_v1"


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

    _pca_results_cache_name = PCA_RESULTS_CACHE_NAME
    _fst_gwss_results_cache_name = FST_GWSS_CACHE_NAME
    _h12_calibration_cache_name = H12_CALIBRATION_CACHE_NAME
    _h12_gwss_cache_name = H12_GWSS_CACHE_NAME
    _g123_gwss_cache_name = G123_GWSS_CACHE_NAME
    _xpehh_gwss_cache_name = XPEHH_GWSS_CACHE_NAME
    _g123_calibration_cache_name = G123_CALIBRATION_CACHE_NAME
    _h1x_gwss_cache_name = H1X_GWSS_CACHE_NAME
    _ihs_gwss_cache_name = IHS_GWSS_CACHE_NAME

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
        tqdm_class=None,
        **storage_options,  # used by fsspec via init_filesystem()
    ):
        super().__init__(
            url=url,
            config_path=CONFIG_PATH,
            cohorts_analysis=cohorts_analysis,
            aim_analysis=None,
            aim_metadata_dtype=None,
            aim_ids=None,
            aim_palettes=None,
            site_filters_analysis=site_filters_analysis,
            default_site_mask="funestus",
            default_phasing_analysis="funestus",
            default_coverage_calls_analysis="funestus",
            bokeh_output_notebook=bokeh_output_notebook,
            results_cache=results_cache,
            log=log,
            debug=debug,
            show_progress=show_progress,
            check_location=check_location,
            pre=pre,
            gcs_url=GCS_URL,
            major_version_number=MAJOR_VERSION_NUMBER,
            major_version_path=MAJOR_VERSION_PATH,
            gff_gene_type="protein_coding_gene",
            gff_default_attributes=("ID", "Parent", "Note", "description"),
            storage_options=storage_options,  # used by fsspec via init_filesystem()
            tqdm_class=tqdm_class,
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
            f"Client location         : {self.client_location}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net. For API documentation see \n"
            f"https://malariagen.github.io/malariagen-data-python/v{malariagen_data.__version__}/Af1.html"
        )
        return text

    def _repr_html_(self):
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
                        See also the <a href="https://malariagen.github.io/malariagen-data-python/v{malariagen_data.__version__}/Af1.html">Af1 API docs</a>.
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
                        <td>{self.client_location}</td>
                    </tr>
                </tbody>
            </table>
        """
        return html

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

    def _view_alignments_add_site_filters_tracks(
        self, *, contig, visibility_window, tracks
    ):
        debug = self._log.debug

        for site_mask in self.site_mask_ids:
            site_filters_vcf_url = f"gs://vo_afun_release/v1.0/site_filters/{self._site_filters_analysis}/vcf/{site_mask}/{contig}_sitefilters.vcf.gz"  # noqa
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
