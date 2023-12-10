import sys

import dask
import dask.array as da
import pandas as pd
import plotly.express as px

import malariagen_data  # used for .__version__

from .anopheles import AnophelesDataResource
from .util import DIM_VARIANT, simple_xarray_concat

# silence dask performance warnings
dask.config.set(**{"array.slicing.split_large_chunks": False})  # type: ignore

MAJOR_VERSION_NUMBER = 3
MAJOR_VERSION_PATH = "v3"
CONFIG_PATH = "v3-config.json"
GCS_URL = "gs://vo_agam_release/"
PCA_RESULTS_CACHE_NAME = "ag3_pca_v1"
FST_GWSS_CACHE_NAME = "ag3_fst_gwss_v1"
H12_CALIBRATION_CACHE_NAME = "ag3_h12_calibration_v1"
H12_GWSS_CACHE_NAME = "ag3_h12_gwss_v1"
G123_CALIBRATION_CACHE_NAME = "ag3_g123_calibration_v1"
G123_GWSS_CACHE_NAME = "ag3_g123_gwss_v1"
XPEHH_GWSS_CACHE_NAME = "ag3_xpehh_gwss_v1"
H1X_GWSS_CACHE_NAME = "ag3_h1x_gwss_v1"
IHS_GWSS_CACHE_NAME = "ag3_ihs_gwss_v1"


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

    virtual_contigs = "2RL", "3RL"
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
        aim_analysis=None,
        site_filters_analysis=None,
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
                "aim_species_gambcolu_arabiensis": object,
                "aim_species_gambiae_coluzzii": object,
                "aim_species": object,
            },
            aim_ids=("gambcolu_vs_arab", "gamb_vs_colu"),
            aim_palettes=AIM_PALETTES,
            site_filters_analysis=site_filters_analysis,
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
            gcs_url=GCS_URL,
            major_version_number=MAJOR_VERSION_NUMBER,
            major_version_path=MAJOR_VERSION_PATH,
            gff_gene_type="gene",
            gff_default_attributes=("ID", "Parent", "Name", "description"),
            storage_options=storage_options,  # used by fsspec via init_filesystem()
            tqdm_class=tqdm_class,
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

    @staticmethod
    def _setup_taxon_colors(plot_kwargs=None):
        import plotly.express as px

        if plot_kwargs is None:
            plot_kwargs = dict()
        taxon_palette = px.colors.qualitative.Vivid
        taxon_color_map = {
            "gambiae": taxon_palette[1],
            "coluzzii": taxon_palette[0],
            "arabiensis": taxon_palette[2],
            "merus": taxon_palette[3],
            "melas": taxon_palette[4],
            "quadriannulatus": taxon_palette[5],
            "fontenillei": taxon_palette[6],
            "gcx1": taxon_palette[7],
            "gcx2": taxon_palette[8],
            "gcx3": taxon_palette[9],
            "gcx4": taxon_palette[10],
            "unassigned": "black",
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
            f"AIM analysis            : {self._aim_analysis}\n"
            f"Site filters analysis   : {self._site_filters_analysis}\n"
            f"Software version        : malariagen_data {malariagen_data.__version__}\n"
            f"Client location         : {self.client_location}\n"
            f"---\n"
            f"Please note that data are subject to terms of use,\n"
            f"for more information see https://www.malariagen.net/data\n"
            f"or contact data@malariagen.net. For API documentation see \n"
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
                        the MalariaGEN website</a> or contact data@malariagen.net.
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

    def _genome_sequence_for_contig(self, *, contig, inline_array, chunks):
        """Obtain the genome sequence for a given contig as an array."""

        if contig in self.virtual_contigs:
            # handle virtual contig with joined arms
            contig_r, contig_l = _chrom_to_contigs(contig)
            d_r = super()._genome_sequence_for_contig(
                contig=contig_r, inline_array=inline_array, chunks=chunks
            )
            d_l = super()._genome_sequence_for_contig(
                contig=contig_l, inline_array=inline_array, chunks=chunks
            )
            return da.concatenate([d_r, d_l])

        return super()._genome_sequence_for_contig(
            contig=contig, inline_array=inline_array, chunks=chunks
        )

    def _genome_features_for_contig(self, *, contig, attributes):
        """Obtain the genome features for a given contig as a pandas DataFrame."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            df_r = super()._genome_features_for_contig(
                contig=contig_r, attributes=attributes
            )
            df_l = super()._genome_features_for_contig(
                contig=contig_l, attributes=attributes
            )
            max_r = super().genome_sequence(region=contig_r).shape[0]
            df_l = df_l.assign(
                start=lambda x: x.start + max_r, end=lambda x: x.end + max_r
            )
            df = pd.concat([df_r, df_l], axis=0)
            df = df.assign(contig=contig)
            return df

        return super()._genome_features_for_contig(contig=contig, attributes=attributes)

    def _snp_genotypes_for_contig(
        self, *, contig, sample_set, field, inline_array, chunks
    ):
        """Access SNP genotypes for a single contig/chromosome and multiple sample sets."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            d_r = super()._snp_genotypes_for_contig(
                contig=contig_r,
                sample_set=sample_set,
                field=field,
                inline_array=inline_array,
                chunks=chunks,
            )
            d_l = super()._snp_genotypes_for_contig(
                contig=contig_l,
                sample_set=sample_set,
                field=field,
                inline_array=inline_array,
                chunks=chunks,
            )
            return da.concatenate([d_r, d_l])

        return super()._snp_genotypes_for_contig(
            contig=contig,
            sample_set=sample_set,
            field=field,
            inline_array=inline_array,
            chunks=chunks,
        )

    def _snp_sites_for_contig(self, contig, *, field, inline_array, chunks):
        """Access SNP sites for a single contig/chromosome."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            field_r = super()._snp_sites_for_contig(
                contig=contig_r, field=field, inline_array=inline_array, chunks=chunks
            )
            field_l = super()._snp_sites_for_contig(
                contig=contig_l, field=field, inline_array=inline_array, chunks=chunks
            )

            if field == "POS":
                max_r = super().genome_sequence(region=contig_r).shape[0]
                field_l = field_l + max_r

            return da.concatenate([field_r, field_l])

        return super()._snp_sites_for_contig(
            contig=contig, field=field, inline_array=inline_array, chunks=chunks
        )

    def _snp_calls_for_contig(self, contig, *, sample_set, inline_array, chunks):
        """Access SNP calls for a single contig/chromosome and a single sample sets as an xarray dataset."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            ds_r = super()._snp_calls_for_contig(
                contig=contig_r,
                sample_set=sample_set,
                inline_array=inline_array,
                chunks=chunks,
            )
            ds_l = super()._snp_calls_for_contig(
                contig=contig_l,
                sample_set=sample_set,
                inline_array=inline_array,
                chunks=chunks,
            )

            ds = simple_xarray_concat([ds_r, ds_l], dim=DIM_VARIANT)

            return ds

        return super()._snp_calls_for_contig(
            contig=contig,
            sample_set=sample_set,
            inline_array=inline_array,
            chunks=chunks,
        )

    def _snp_variants_for_contig(self, contig, *, inline_array, chunks):
        """Access SNP variants for a single contig/chromosome as an xarray dataset."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            ds_r = super()._snp_variants_for_contig(
                contig=contig_r,
                inline_array=inline_array,
                chunks=chunks,
            )
            ds_l = super()._snp_variants_for_contig(
                contig=contig_l,
                inline_array=inline_array,
                chunks=chunks,
            )
            max_r = super().genome_sequence(region=contig_r).shape[0]
            ds_l["variant_position"] = ds_l["variant_position"] + max_r

            ds = simple_xarray_concat([ds_r, ds_l], dim=DIM_VARIANT)

            return ds

        return super()._snp_variants_for_contig(
            contig=contig,
            inline_array=inline_array,
            chunks=chunks,
        )

    def _haplotype_sites_for_contig(
        self, *, contig, analysis, field, inline_array, chunks
    ):
        """Access haplotype site data for a single contig/chromosome."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            pos_r = super()._haplotype_sites_for_contig(
                contig=contig_r,
                analysis=analysis,
                field=field,
                inline_array=inline_array,
                chunks=chunks,
            )
            pos_l = super()._haplotype_sites_for_contig(
                contig=contig_l,
                analysis=analysis,
                field=field,
                inline_array=inline_array,
                chunks=chunks,
            )
            max_r = super().genome_sequence(region=contig_r).shape[0]
            pos_l = pos_l + max_r

            return da.concatenate([pos_r, pos_l])

        return super()._haplotype_sites_for_contig(
            contig=contig,
            analysis=analysis,
            field=field,
            inline_array=inline_array,
            chunks=chunks,
        )

    def _haplotypes_for_contig(
        self, *, contig, sample_set, analysis, inline_array, chunks
    ):
        """Access haplotypes for a single whole chromosome and a single sample sets."""

        if contig in self.virtual_contigs:
            contig_r, contig_l = _chrom_to_contigs(contig)

            ds_r = super()._haplotypes_for_contig(
                contig=contig_r,
                sample_set=sample_set,
                analysis=analysis,
                inline_array=inline_array,
                chunks=chunks,
            )
            ds_l = super()._haplotypes_for_contig(
                contig=contig_l,
                sample_set=sample_set,
                analysis=analysis,
                inline_array=inline_array,
                chunks=chunks,
            )

            # handle case where no haplotypes available for given sample set
            # then convert genome coordinates
            if ds_l is not None:
                max_r = super().genome_sequence(region=contig_r).shape[0]
                ds_l["variant_position"] = ds_l["variant_position"] + max_r
                ds = simple_xarray_concat([ds_r, ds_l], dim=DIM_VARIANT)
                return ds

            return None

        return super()._haplotypes_for_contig(
            contig=contig,
            sample_set=sample_set,
            analysis=analysis,
            inline_array=inline_array,
            chunks=chunks,
        )

    def _view_alignments_add_site_filters_tracks(
        self, *, contig, visibility_window, tracks
    ):
        debug = self._log.debug

        for site_mask in self.site_mask_ids:
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
        # override parent class to add AIM analysis
        params["aim_analysis"] = self._aim_analysis


def _chrom_to_contigs(chrom):
    """Split a chromosome name into two contig names."""
    assert chrom in ["2RL", "3RL"]
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    return contig_r, contig_l
