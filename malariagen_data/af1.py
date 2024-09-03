import sys

import plotly.express as px  # type: ignore

import malariagen_data
from .anopheles import AnophelesDataResource

MAJOR_VERSION_NUMBER = 1
MAJOR_VERSION_PATH = "v1.0"
CONFIG_PATH = "v1.0-config.json"
GCS_DEFAULT_URL = "gs://vo_afun_release_master_us_central1/"
GCS_REGION_URLS = {
    "us-central1": "gs://vo_afun_release_master_us_central1",
}
XPEHH_GWSS_CACHE_NAME = "af1_xpehh_gwss_v1"
IHS_GWSS_CACHE_NAME = "af1_ihs_gwss_v1"

TAXON_PALETTE = px.colors.qualitative.Plotly
TAXON_COLORS = {
    "funestus": TAXON_PALETTE[0],
}


class Af1(AnophelesDataResource):
    """Provides access to data from Af1.x releases.

    Parameters
    ----------
    url : str, optional
        Base path to data. Defaults to use Google Cloud Storage, or can
        be a local path on your file system if data have been downloaded.
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

    _xpehh_gwss_cache_name = XPEHH_GWSS_CACHE_NAME
    _ihs_gwss_cache_name = IHS_GWSS_CACHE_NAME

    def __init__(
        self,
        url=None,
        bokeh_output_notebook=True,
        results_cache=None,
        log=sys.stdout,
        debug=False,
        show_progress=True,
        check_location=True,
        cohorts_analysis=None,
        site_filters_analysis=None,
        discordant_read_calls_analysis=None,
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
            discordant_read_calls_analysis=discordant_read_calls_analysis,
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
            gcs_default_url=GCS_DEFAULT_URL,
            gcs_region_urls=GCS_REGION_URLS,
            major_version_number=MAJOR_VERSION_NUMBER,
            major_version_path=MAJOR_VERSION_PATH,
            gff_gene_type="protein_coding_gene",
            gff_gene_name_attribute="Note",
            gff_default_attributes=("ID", "Parent", "Note", "description"),
            storage_options=storage_options,  # used by fsspec via init_filesystem()
            tqdm_class=tqdm_class,
            taxon_colors=TAXON_COLORS,
            virtual_contigs=None,
            gene_names=None,
        )

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
            f"or contact support@malariagen.net. For API documentation see \n"
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
                        the MalariaGEN website</a> or contact support@malariagen.net.
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
