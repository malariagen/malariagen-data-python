import sys

import allel
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from .anopheles import Anopheles

try:
    # noinspection PyPackageRequirements
    from google import colab
except ImportError:
    colab = None

import malariagen_data  # used for .__version__

from .util import DIM_ALLELE, DIM_VARIANT, CacheMiss, da_from_zarr, hash_params

MAJOR_VERSION_INT = 1
MAJOR_VERSION_GCS_STR = "v1.0"
PUBLIC_RELEASES = ("1.0",)
GCS_URL = "gs://vo_afun_release/"
GENESET_GFF3_PATH = "reference/genome/idAnoFuneDA-416_04/GCF_943734845.2_idAnoFuneDA-416_04_genomic.gff3.gz"
GENOME_FASTA_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa"
)
GENOME_FAI_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.fa.fai"
)
GENOME_ZARR_PATH = (
    "reference/genome/idAnoFuneDA-416_04/idAnoFuneDA-416_04_1.curated_primary.zarr"
)
GENOME_REF_ID = "idAnoFuneDA-416_04"
GENOME_REF_NAME = "Anopheles funestus"

CONTIGS = "2RL", "3RL", "X"
DEFAULT_SITE_FILTERS_ANALYSIS = "dt_20200416"

DEFAULT_GENOME_PLOT_WIDTH = 800  # width in px for bokeh genome plots
DEFAULT_GENES_TRACK_HEIGHT = 120  # height in px for bokeh genes track plots

PCA_RESULTS_CACHE_NAME = "af1_pca_v1"
SNP_ALLELE_COUNTS_CACHE_NAME = "af1_snp_allele_counts_v1"
DEFAULT_SITE_MASK = "funestus"


class Af1(Anopheles):
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
        ...     "simplecache::gs://vo_afub_release",
        ...     simplecache=dict(cache_storage="gcs_cache"),
        ... )

    Set up caching of some longer-running computations on the local file system,
    in a directory named "results_cache":

        >>> af1 = malariagen_data.Af1(results_cache="results_cache")

    """

    contigs = CONTIGS
    _public_releases = PUBLIC_RELEASES
    _major_version_int = MAJOR_VERSION_INT
    _major_version_gcs_str = MAJOR_VERSION_GCS_STR
    _geneset_gff3_path = GENESET_GFF3_PATH
    _genome_fasta_path = GENOME_FASTA_PATH
    _genome_fai_path = GENOME_FAI_PATH
    _genome_zarr_path = GENOME_ZARR_PATH
    _genome_ref_id = GENOME_REF_ID
    _genome_ref_name = GENOME_REF_NAME
    _gcs_url = GCS_URL

    def __init__(
        self,
        url=GCS_URL,
        site_filters_analysis=DEFAULT_SITE_FILTERS_ANALYSIS,
        bokeh_output_notebook=True,
        results_cache=None,
        log=sys.stdout,
        debug=False,
        show_progress=True,
        check_location=True,
        pre=False,
        **kwargs,  # used by simplecache, init_filesystem(url, **kwargs)
    ):

        super().__init__(
            url=url,
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

    def __repr__(self):
        text = (
            f"<MalariaGEN Af1 API client>\n"
            f"Storage URL             : {self._url}\n"
            f"Data releases available : {', '.join(self.releases)}\n"
            f"Results cache           : {self._results_cache}\n"
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

    # TODO: generalise (species, cohorts) so we can abstract to parent class
    def _sample_metadata(self, *, sample_set):
        df = self._read_general_metadata(sample_set=sample_set)
        # TODO: when cohorts are available
        # df_cohorts = self._read_cohort_metadata(sample_set=sample_set)
        # df = df.merge(df_cohorts, on="sample_id", sort=False)
        return df

    def _transcript_to_gene_name(self, transcript):
        df_geneset = self.geneset().set_index("ID")
        rec_transcript = df_geneset.loc[transcript]
        parent = rec_transcript["Parent"]
        rec_parent = df_geneset.loc[parent]

        # TODO: tailor to Af
        # manual overrides
        # if parent == "AGAP004707":
        #     parent_name = "Vgsc/para"
        # else:
        #     parent_name = rec_parent["Name"]
        parent_name = rec_parent["Name"]

        return parent_name

    def _site_mask_ids(self):
        if self._site_filters_analysis == "dt_20200416":
            return ["funestus"]
        elif self._site_filters_analysis == "sc_20220908":
            return ["funestus"]
        else:
            raise ValueError

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
        release_path = self._release_to_path(release=release)

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

    def results_cache_get(self, *, name, params):
        debug = self._log.debug
        if self._results_cache is None:
            raise CacheMiss
        params = params.copy()
        # TODO: when cohorts are available
        # params["cohorts_analysis"] = self._cohorts_analysis
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
        # TODO: when cohorts are available
        # params["cohorts_analysis"] = self._cohorts_analysis
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
    ):
        """Compute SNP allele counts. This returns the number of times each
        SNP allele was observed in the selected samples.

        Parameters
        ----------
        region : str or Region
            Contig name (e.g., "2RL"), gene name (e.g., "gene-LOC125762289") or
            genomic region defined with coordinates (e.g.,
            "2RL:44989425-44998059").
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "1229-VO-GH-DADZIE-VMF00095") or a list of
            sample set identifiers (e.g., ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"]) or a
            release identifier (e.g., "1.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".
        site_mask : {"funestus"}
            Site filters mask to apply.

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
        name = SNP_ALLELE_COUNTS_CACHE_NAME

        # normalize params for consistent hash value
        params = dict(
            region=self.resolve_region(region).to_dict(),
            sample_sets=self._prep_sample_sets_arg(sample_sets=sample_sets),
            sample_query=sample_query,
            site_mask=site_mask,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._snp_allele_counts(**params)
            self.results_cache_set(name=name, params=params, results=results)

        ac = results["ac"]
        return ac

    def pca(
        self,
        region,
        n_snps,
        thin_offset=0,
        sample_sets=None,
        sample_query=None,
        site_mask=DEFAULT_SITE_MASK,
        min_minor_ac=2,
        max_missing_an=0,
        n_components=20,
    ):
        """Run a principal components analysis (PCA) using biallelic SNPs from
        the selected genome region and samples.

        Parameters
        ----------
        region : str
            Contig name (e.g., "2RL"), gene name (e.g., "gene-LOC125762289") or
            genomic region defined with coordinates (e.g.,
            "2RL:44989425-44998059").
        n_snps : int
            The desired number of SNPs to use when running the analysis.
            SNPs will be evenly thinned to approximately this number.
        thin_offset : int, optional
            Starting index for SNP thinning. Change this to repeat the analysis
            using a different set of SNPs.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "1229-VO-GH-DADZIE-VMF00095") or a list of
            sample set identifiers (e.g., ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"]) or a
            release identifier (e.g., "1.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "country == 'Burkina Faso'".
        site_mask : {"funestus"}
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
        name = PCA_RESULTS_CACHE_NAME

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

    def plot_snps_track(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        site_mask=DEFAULT_SITE_MASK,
        width=800,
        height=120,
        max_snps=200_000,
        show=True,
    ):
        # TODO docstring
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
            "pass_funestus": ds_sites["variant_filter_pass_funestus"].values,
        }
        data = pd.DataFrame(cols)

        debug("create figure")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        pos = data["pos"].values
        x_min = pos[0]
        x_max = pos[-1]
        x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        tooltips = [
            ("Position", "$x{0,0}"),
            (
                "Alleles",
                "@allele_0 (@ac_0), @allele_1 (@ac_1), @allele_2 (@ac_2), @allele_3 (@ac_3)",
            ),
            ("No. alleles", "@allelism"),
            ("Allele calls", "@an"),
            ("Pass funestus", "@pass_funestus"),
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
        site_mask=DEFAULT_SITE_MASK,
        width=800,
        track_height=80,
        genes_height=120,
        show=True,
    ):
        # TODO docstring
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

    @staticmethod
    def _setup_taxon_colors(plot_kwargs):
        import plotly.express as px

        taxon_palette = px.colors.qualitative.Plotly
        taxon_color_map = {
            "funestus": taxon_palette[0],
            "gcx1": taxon_palette[1],
            "gcx2": taxon_palette[2],
            "gcx3": taxon_palette[3],
        }
        plot_kwargs["color_discrete_map"] = taxon_color_map
        plot_kwargs["category_orders"] = {"taxon": list(taxon_color_map.keys())}
