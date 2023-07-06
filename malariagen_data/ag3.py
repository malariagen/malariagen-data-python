import sys
import warnings
from bisect import bisect_left, bisect_right
from typing import List

import dask
import dask.array as da
import numba
import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr

import malariagen_data  # used for .__version__

from .anoph import base_params
from .anopheles import AnophelesDataResource
from .util import (
    DIM_VARIANT,
    Region,
    parse_multi_region,
    region_str,
    simple_xarray_concat,
)

# silence dask performance warnings
dask.config.set(**{"array.slicing.split_large_chunks": False})  # type: ignore

MAJOR_VERSION_NUMBER = 3
MAJOR_VERSION_PATH = "v3"
CONFIG_PATH = "v3-config.json"
GCS_URL = "gs://vo_agam_release/"
DEFAULT_MAX_COVERAGE_VARIANCE = 0.2
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
            f"AIM analysis            : {self._aim_analysis}\n"
            f"Site filters analysis   : {self._site_filters_analysis}\n"
            f"Software version        : malariagen_data {malariagen_data.__version__}\n"
            f"Client location         : {self.client_location}\n"
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

    def gene_cnv(
        self,
        region: base_params.regions,
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

        regions: List[Region] = parse_multi_region(self, region)
        del region

        ds = simple_xarray_concat(
            [
                self._gene_cnv(
                    region=r,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    max_coverage_variance=max_coverage_variance,
                )
                for r in regions
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
        region: base_params.regions,
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
        regions: List[Region] = parse_multi_region(self, region)
        del region

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
                for r in regions
            ],
            axis=0,
        )

        debug("add metadata")
        title = f"Gene CNV frequencies ({region_str(regions)})"
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

    def gene_cnv_frequencies_advanced(
        self,
        region: base_params.regions,
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

        regions: List[Region] = parse_multi_region(self, region)
        del region

        ds = simple_xarray_concat(
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
                for r in regions
            ],
            dim="variants",
        )

        title = f"Gene CNV frequencies ({region_str(regions)})"
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


def _chrom_to_contigs(chrom):
    """Split a chromosome name into two contig names."""
    assert chrom in ["2RL", "3RL"]
    contig_r = chrom[0] + chrom[1]
    contig_l = chrom[0] + chrom[2]
    return contig_r, contig_l
