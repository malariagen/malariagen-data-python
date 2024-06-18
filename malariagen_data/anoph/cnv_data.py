from typing import Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr  # type: ignore
from numpydoc_decorator import doc  # type: ignore

from ..util import (
    DIM_SAMPLE,
    DIM_VARIANT,
    Region,
    check_types,
    da_from_zarr,
    init_zarr_store,
    parse_multi_region,
    parse_single_region,
    simple_xarray_concat,
)
from . import base_params, cnv_params, gplt_params
from .genome_features import AnophelesGenomeFeaturesData
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata


class AnophelesCnvData(
    AnophelesSampleMetadata, AnophelesGenomeFeaturesData, AnophelesGenomeSequenceData
):
    def __init__(
        self,
        discordant_read_calls_analysis: Optional[str] = None,
        default_coverage_calls_analysis: Optional[str] = None,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # If provided, this analysis version will override the
        # default value provided in the release configuration.
        self._discordant_read_calls_analysis_override = discordant_read_calls_analysis

        # These will vary between data resources.
        self._default_coverage_calls_analysis = default_coverage_calls_analysis

        # set up caches
        self._cache_cnv_hmm: Dict = dict()
        self._cache_cnv_coverage_calls: Dict = dict()
        self._cache_cnv_discordant_read_calls: Dict = dict()

    @property
    def _discordant_read_calls_analysis(self) -> Optional[str]:
        if isinstance(self._discordant_read_calls_analysis_override, str):
            return self._discordant_read_calls_analysis_override
        else:
            # N.B., this will return None if the key is not present in the
            # config.
            return self.config.get("DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS")

    @property
    def coverage_calls_analysis_ids(self) -> Tuple[str, ...]:
        """Identifiers for the different coverage calls analyses that are available.
        These are values than can be used for the `coverage_calls_analysis` parameter in any
        method making using of CNV data.

        """
        return tuple(self.config.get("COVERAGE_CALLS_ANALYSIS_IDS", ()))  # ensure tuple

    @check_types
    @doc(
        summary="Open CNV HMM zarr.",
        returns="Zarr hierarchy.",
    )
    def open_cnv_hmm(self, sample_set: base_params.sample_set) -> zarr.hierarchy.Group:
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

    @check_types
    @doc(
        summary="Access CNV HMM data from CNV calling.",
        returns="An xarray dataset of CNV HMM calls and associated data.",
    )
    def cnv_hmm(
        self,
        region: base_params.regions,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> xr.Dataset:
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
        regions: List[Region] = parse_multi_region(self, region)
        del region

        with self._spinner("Access CNV HMM data"):
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
                    loc_region = index.overlaps(other)  # type: ignore
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

    @check_types
    @doc(
        summary="Open CNV coverage calls zarr.",
        returns="Zarr hierarchy.",
    )
    def open_cnv_coverage_calls(
        self,
        sample_set: base_params.sample_set,
        analysis: cnv_params.coverage_calls_analysis,
    ) -> zarr.hierarchy.Group:
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
                    f"CNV coverage calls analysis f{analysis!r} not implemented for sample set {sample_set!r}"
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

    @check_types
    @doc(
        summary="Access CNV HMM data from genome-wide CNV discovery and filtering.",
        returns="An xarray dataset of CNV alleles and genotypes.",
    )
    def cnv_coverage_calls(
        self,
        region: base_params.regions,
        sample_set: base_params.sample_set,
        analysis: cnv_params.coverage_calls_analysis,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> xr.Dataset:
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
                loc_region = index.overlaps(other)  # type: ignore
                x = x.isel(variants=loc_region)

            lx.append(x)
        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

        return ds

    @check_types
    @doc(
        summary="Open CNV discordant read calls zarr.",
        returns="Zarr hierarchy.",
    )
    def open_cnv_discordant_read_calls(
        self, sample_set: base_params.sample_set
    ) -> zarr.hierarchy.Group:
        try:
            return self._cache_cnv_discordant_read_calls[sample_set]
        except KeyError:
            release = self.lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
            analysis = self._discordant_read_calls_analysis
            if analysis:
                calls_version = f"discordant_read_calls_{analysis}"
            else:
                calls_version = "discordant_read_calls"
            path = f"{self._base_path}/{release_path}/cnv/{sample_set}/{calls_version}/zarr"
            # print(analysis)
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

    @check_types
    @doc(
        summary="Access CNV discordant read calls data.",
        returns="An xarray dataset of CNV alleles and genotypes.",
    )
    def cnv_discordant_read_calls(
        self,
        contig: base_params.contigs,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> xr.Dataset:
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

        return ds

    @check_types
    @doc(
        summary="Plot CNV HMM data for a single sample, using bokeh.",
        returns="Bokeh figure.",
        parameters=dict(
            y_max="Y axis limit or 'auto'.",
        ),
    )
    def plot_cnv_hmm_coverage_track(
        self,
        sample: base_params.samples,
        region: base_params.region,
        sample_set: Optional[base_params.sample_set] = None,
        y_max: Union[float, str] = "auto",
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        circle_kwargs: Optional[gplt_params.circle_kwargs] = None,
        line_kwargs: Optional[gplt_params.line_kwargs] = None,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
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
            y_max_float = data["call_CN"].max() + 2
        else:
            y_max_float = y_max

        debug("set up x range")
        x_min = data["variant_position"].values[0]
        x_max = data["variant_end"].values[-1]
        if x_range is None:
            x_range = bkmod.Range1d(x_min, x_max, bounds="auto")

        debug("create a figure for plotting")
        xwheel_zoom = bkmod.WheelZoomTool(dimensions="width", maintain_focus=False)
        fig = bkplt.figure(
            title=f"CNV HMM - {sample_id} ({sample_set})",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset", "save"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, y_max_float),
            output_backend=output_backend,
        )

        debug("plot the normalised coverage data")
        circle_kwargs_mutable = dict(circle_kwargs) if circle_kwargs else {}
        circle_kwargs_mutable["size"] = circle_kwargs_mutable.get("size", 3)
        circle_kwargs_mutable["line_width"] = circle_kwargs_mutable.get("line_width", 1)
        circle_kwargs_mutable["line_color"] = circle_kwargs_mutable.get(
            "line_color", "black"
        )
        circle_kwargs_mutable["fill_color"] = circle_kwargs_mutable.get(
            "fill_color", None
        )
        circle_kwargs_mutable["legend_label"] = circle_kwargs_mutable.get(
            "legend_label", "Coverage"
        )
        fig.scatter(
            x="variant_midpoint",
            y="call_NormCov",
            source=data,
            marker="circle",
            **circle_kwargs_mutable,
        )

        debug("plot the HMM state")
        line_kwargs_mutable = dict(line_kwargs) if line_kwargs else {}
        line_kwargs_mutable["width"] = line_kwargs_mutable.get("width", 2)
        line_kwargs_mutable["legend_label"] = line_kwargs_mutable.get(
            "legend_label", "HMM"
        )
        fig.line(x="variant_midpoint", y="call_CN", source=data, **line_kwargs_mutable)

        debug("tidy up the plot")
        fig.yaxis.axis_label = "Copy number"
        fig.yaxis.ticker = list(range(int(y_max_float) + 1))
        self._bokeh_style_genome_xaxis(fig, region_prepped.contig)
        fig.add_layout(fig.legend[0], "right")

        if show:
            bkplt.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Plot CNV HMM data for a single sample, together with a genes track, using bokeh.",
        returns="Bokeh figure.",
        parameters=dict(
            y_max="Y axis limit or 'auto'.",
        ),
    )
    def plot_cnv_hmm_coverage(
        self,
        sample: base_params.samples,
        region: base_params.region,
        sample_set: Optional[base_params.sample_set] = None,
        y_max: Union[float, str] = "auto",
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.track_height = 170,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        circle_kwargs: Optional[gplt_params.circle_kwargs] = None,
        line_kwargs: Optional[gplt_params.line_kwargs] = None,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
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

    @check_types
    @doc(
        summary="Plot CNV HMM data for multiple samples as a heatmap, using bokeh.",
        returns="Bokeh figure.",
    )
    def plot_cnv_hmm_heatmap_track(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        row_height: gplt_params.row_height = 7,
        height: Optional[gplt_params.height] = None,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
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
            if isinstance(sample_sets, str):
                sample_sets_text = sample_sets
            else:
                sample_sets_text = ", ".join(sample_sets)
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
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset", "save"],
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

    @check_types
    @doc(
        summary="Plot CNV HMM data for multiple samples as a heatmap, with a genes track, using bokeh.",
        returns="Bokeh figure.",
    )
    def plot_cnv_hmm_heatmap(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        row_height: gplt_params.row_height = 7,
        track_height: Optional[gplt_params.track_height] = None,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
    ) -> gplt_params.figure:
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
