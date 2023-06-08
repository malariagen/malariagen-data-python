from typing import List

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr

from ..util import (
    DIM_SAMPLE,
    DIM_VARIANT,
    Region,
    da_from_zarr,
    init_zarr_store,
    parse_multi_region,
    parse_single_region,
    simple_xarray_concat,
)
from . import base_params, gplt_params
from .genome_features import AnophelesGenomeFeaturesData
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata

DEFAULT_MAX_COVERAGE_VARIANCE = 0.2


class AnophelesCnvData(
    AnophelesSampleMetadata, AnophelesGenomeFeaturesData, AnophelesGenomeSequenceData
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # set up caches
        self._cache_cnv_hmm = dict()
        self._cache_cnv_coverage_calls = dict()
        self._cache_cnv_discordant_read_calls = dict()

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
            release = self.lookup_release(sample_set=sample_set)
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
        region: base_params.region,
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
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
        regions: List[Region] = parse_multi_region(self, region)
        del region

        debug("access CNV HMM data and concatenate as needed")
        lx = []
        for r in regions:
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
            x = simple_xarray_concat(ly, dim=DIM_SAMPLE)

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
        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

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
            release = self.lookup_release(sample_set=sample_set)
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
        region: base_params.region,
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
        regions: List[Region] = parse_multi_region(self, region)
        del region

        debug("access data and concatenate as needed")
        lx = []
        for r in regions:
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
        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

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
            release = self.lookup_release(sample_set=sample_set)
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
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
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

            x = simple_xarray_concat(ly, dim=DIM_SAMPLE)
            lx.append(x)

        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    def plot_cnv_hmm_coverage_track(
        self,
        sample,
        region: base_params.single_region,
        sample_set=None,
        y_max="auto",
        sizing_mode=gplt_params.sizing_mode_default,
        width=gplt_params.width_default,
        height=200,
        circle_kwargs=None,
        line_kwargs=None,
        show=True,
        x_range=None,
        output_backend="webgl",
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
        x_range : bokeh.models.Range, optional
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
        region_prepped: Region = parse_single_region(self, region)
        del region

        debug("access sample metadata, look up sample")
        sample_rec = self.lookup_sample(sample=sample, sample_set=sample_set)
        sample_id = sample_rec.name  # sample_id
        sample_set = sample_rec["sample_set"]

        debug("access HMM data")
        hmm = self.cnv_hmm(
            region=region_prepped, sample_sets=sample_set, max_coverage_variance=None
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
            output_backend=output_backend,
        )

        debug("plot the normalised coverage data")
        if circle_kwargs is None:
            circle_kwargs = dict()
        circle_kwargs.setdefault("size", 3)
        circle_kwargs.setdefault("line_width", 1)
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
        self._bokeh_style_genome_xaxis(fig, region_prepped.contig)
        fig.add_layout(fig.legend[0], "right")

        if show:
            bkplt.show(fig)
            return None
        else:
            return fig

    def plot_cnv_hmm_coverage(
        self,
        sample,
        region,
        sample_set=None,
        y_max="auto",
        sizing_mode=gplt_params.sizing_mode_default,
        width=gplt_params.width_default,
        track_height=170,
        genes_height=gplt_params.genes_height_default,
        circle_kwargs=None,
        line_kwargs=None,
        show=True,
        output_backend="webgl",
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
            output_backend=output_backend,
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
            output_backend=output_backend,
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
            return None
        else:
            return fig

    def plot_cnv_hmm_heatmap_track(
        self,
        region: base_params.single_region,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        sizing_mode=gplt_params.sizing_mode_default,
        width=gplt_params.width_default,
        row_height=7,
        height=None,
        show=True,
        output_backend="webgl",
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

        region_prepped: Region = parse_single_region(self, region)
        del region

        debug("access HMM data")
        ds_cnv = self.cnv_hmm(
            region=region_prepped,
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
            output_backend=output_backend,
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
        self._bokeh_style_genome_xaxis(fig, region_prepped.contig)
        fig.yaxis.ticker = bkmod.FixedTicker(
            ticks=np.arange(len(sample_id)),
        )
        fig.yaxis.major_label_overrides = {i: s for i, s in enumerate(sample_id)}
        fig.yaxis.major_label_text_font_size = f"{row_height}px"

        debug("add color bar")
        # For some reason, mypy reports: Module has no attribute "ColorBar"
        # ...but this works fine, so ignore for now.
        color_bar = bkmod.ColorBar(  # type: ignore
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
            return None
        else:
            return fig

    def plot_cnv_hmm_heatmap(
        self,
        region,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        sizing_mode=gplt_params.sizing_mode_default,
        width=gplt_params.width_default,
        row_height=7,
        track_height=None,
        genes_height=gplt_params.genes_height_default,
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
            return None
        else:
            return fig
