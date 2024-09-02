import sys

import dask
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore

import malariagen_data
from .anopheles import AnophelesDataResource

# silence dask performance warnings
dask.config.set(**{"array.slicing.split_large_chunks": False})  # type: ignore

MAJOR_VERSION_NUMBER = 3
MAJOR_VERSION_PATH = "v3"
CONFIG_PATH = "v3-config.json"
GCS_DEFAULT_URL = "gs://vo_agam_release_master_us_central1/"
GCS_REGION_URLS = {
    "us-central1": "gs://vo_agam_release_master_us_central1",
}
XPEHH_GWSS_CACHE_NAME = "ag3_xpehh_gwss_v1"
IHS_GWSS_CACHE_NAME = "ag3_ihs_gwss_v1"
VIRTUAL_CONTIGS = {
    "2RL": ("2R", "2L"),
    "3RL": ("3R", "3L"),
    "23X": ("2R", "2L", "3R", "3L", "X"),
}
GENE_NAMES = {
    "AGAP004707": "Vgsc/para",
}


def _setup_aim_palettes():
    # Set up default AIMs color palettes.
    colors = px.colors.qualitative.T10
    color_gambcolu = colors[6]
    color_gambcolu_arab_het = colors[5]
    color_arab = colors[4]
    color_gamb = colors[0]
    color_gamb_colu_het = colors[5]
    color_colu = colors[2]
    color_missing = "white"
    aim_palettes = {
        "gambcolu_vs_arab": (
            color_missing,
            color_gambcolu,
            color_gambcolu_arab_het,
            color_arab,
        ),
        "gamb_vs_colu": (
            color_missing,
            color_gamb,
            color_gamb_colu_het,
            color_colu,
        ),
    }
    return aim_palettes


AIM_PALETTES = _setup_aim_palettes()

TAXON_PALETTE = px.colors.qualitative.Vivid
TAXON_COLORS = {
    "gambiae": TAXON_PALETTE[1],
    "coluzzii": TAXON_PALETTE[0],
    "arabiensis": TAXON_PALETTE[2],
    "merus": TAXON_PALETTE[3],
    "melas": TAXON_PALETTE[4],
    "quadriannulatus": TAXON_PALETTE[5],
    "fontenillei": TAXON_PALETTE[6],
    "gcx1": TAXON_PALETTE[7],
    "gcx2": TAXON_PALETTE[8],
    "gcx3": TAXON_PALETTE[9],
    "gcx4": TAXON_PALETTE[10],
    "unassigned": "black",
}


class Ag3(AnophelesDataResource):
    """Provides access to data from Ag3.x releases.

    Parameters
    ----------
    url : str, optional
        Base path to data. Defaults to use Google Cloud Storage, or can
        be a local path on your file system if data have been downloaded.
    cohorts_analysis : str, optional
        Cohort analysis version.
    aim_analysis : str, optional
        AIM analysis version.
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
        aim_analysis=None,
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
            aim_analysis=aim_analysis,
            aim_metadata_dtype={
                "aim_species_fraction_arab": "float64",
                "aim_species_fraction_colu": "float64",
                "aim_species_fraction_colu_no2l": "float64",
                "aim_species_gambcolu_arabiensis": "object",
                "aim_species_gambiae_coluzzii": "object",
                "aim_species": "object",
            },
            aim_ids=("gambcolu_vs_arab", "gamb_vs_colu"),
            aim_palettes=AIM_PALETTES,
            site_filters_analysis=site_filters_analysis,
            discordant_read_calls_analysis=discordant_read_calls_analysis,
            default_site_mask="gamb_colu_arab",
            default_phasing_analysis="gamb_colu_arab",
            default_coverage_calls_analysis="gamb_colu",
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
            gff_gene_type="gene",
            gff_gene_name_attribute="Name",
            gff_default_attributes=("ID", "Parent", "Name", "description"),
            storage_options=storage_options,  # used by fsspec via init_filesystem()
            tqdm_class=tqdm_class,
            taxon_colors=TAXON_COLORS,
            virtual_contigs=VIRTUAL_CONTIGS,
            gene_names=GENE_NAMES,
        )

        # set up caches
        self._cache_cross_metadata = None

    @property
    def v3_wild(self):
        """Legacy, convenience property to access sample sets from the
        3.0 release, excluding the lab crosses."""
        return [
            x
            for x in self.sample_sets(release="3.0")["sample_set"].tolist()
            if x != "AG1000G-X"
        ]

    def __repr__(self):
        text = (
            f"<MalariaGEN Ag3 API client>\n"
            f"Storage URL             : {self._url}\n"
            f"Data releases available : {', '.join(self.releases)}\n"
            f"Results cache           : {self._results_cache}\n"
            f"Cohorts analysis        : {self._cohorts_analysis}\n"
            f"AIM analysis            : {self._aim_analysis}\n"
            f"Site filters analysis   : {self._site_filters_analysis}\n"
            f"Software version        : malariagen_data {malariagen_data.__version__}\n"
            f"Client location         : {self.client_location}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact support@malariagen.net. For API documentation see \n"
            f"https://malariagen.github.io/malariagen-data-python/v{malariagen_data.__version__}/Ag3.html"
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
                        the MalariaGEN website</a> or contact support@malariagen.net.
                        See also the <a href="https://malariagen.github.io/malariagen-data-python/v{malariagen_data.__version__}/Ag3.html">Ag3 API docs</a>.
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
                            AIM analysis
                        </th>
                        <td>{self._aim_analysis}</td>
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

    def _results_cache_add_analysis_params(self, params):
        super()._results_cache_add_analysis_params(params)
        # override parent class to add AIM analysis
        params["aim_analysis"] = self._aim_analysis
