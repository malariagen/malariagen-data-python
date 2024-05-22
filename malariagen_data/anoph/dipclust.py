from typing import Optional, Tuple

import allel  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore

from ..util import (
    CacheMiss,
    check_types,
    multiallelic_diplotype_pdist,
    multiallelic_diplotype_mean_sqeuclidean,
    multiallelic_diplotype_mean_cityblock,
)
from ..plotly_dendrogram import plot_dendrogram
from . import base_params, plotly_params, tree_params, dipclust_params
from .base_params import DEFAULT
from .snp_frq import AnophelesSnpFrequencyAnalysis
import pandas as pd


class AnophelesDipClustAnalysis(AnophelesSnpFrequencyAnalysis):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @check_types
    @doc(
        summary="Hierarchically cluster diplotypes in region and produce an interactive plot.",
        parameters=dict(
            leaf_y="Y coordinate at which to plot the leaf markers.",
            return_order_dict="Return a dictionary containing the order of samples in the dendrogram.",
        ),
    )
    def plot_diplotype_clustering(
        self,
        region: base_params.regions,
        site_mask: base_params.site_mask = DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        linkage_method: dipclust_params.linkage_method = dipclust_params.linkage_method_default,
        distance_metric: dipclust_params.distance_metric = dipclust_params.distance_metric_default,
        count_sort: Optional[tree_params.count_sort] = None,
        distance_sort: Optional[tree_params.distance_sort] = None,
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        width: plotly_params.width = None,
        height: plotly_params.height = 500,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        leaf_y: int = 0,
        marker_size: plotly_params.marker_size = 5,
        line_width: plotly_params.line_width = 0.5,
        line_color: plotly_params.line_color = "black",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
        return_order_dict: bool = False,
    ) -> plotly_params.figure:
        import sys

        debug = self._log.debug

        # Normalise params.
        if count_sort is None and distance_sort is None:
            count_sort = True
            distance_sort = False

        # This is needed to avoid RecursionError on some haplotype clustering analyses
        # with larger numbers of haplotypes.
        sys.setrecursionlimit(10_000)

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        dist, gt_samples, n_snps_used = self.diplotype_pairwise_distances(
            region=region,
            site_mask=site_mask,
            sample_sets=sample_sets,
            sample_query=sample_query,
            cohort_size=cohort_size,
            distance_metric=distance_metric,
            random_seed=random_seed,
        )

        # Align sample metadata with genotypes.
        df_samples = (
            df_samples.set_index("sample_id").loc[gt_samples.tolist()].reset_index()
        )

        # Normalise color and symbol parameters.
        symbol_prepped = self._setup_sample_symbol(
            data=df_samples,
            symbol=symbol,
        )
        del symbol
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=df_samples,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence

        # Configure hover data.
        hover_data = self._setup_sample_hover_data_plotly(
            color=color_prepped, symbol=symbol_prepped
        )

        # Construct plot title.
        if title is True:
            title_lines = []
            if sample_sets is not None:
                title_lines.append(f"Sample sets: {sample_sets}")
            if sample_query is not None:
                title_lines.append(f"Sample query: {sample_query}")
            title_lines.append(f"Genomic region: {region} ({n_snps_used:,} SNPs)")
            title = "<br>".join(title_lines)

        # Create the plot.
        with self._spinner("Plot dendrogram"):
            fig = plot_dendrogram(
                dist=dist,
                linkage_method=linkage_method,
                count_sort=count_sort,
                distance_sort=distance_sort,
                render_mode=render_mode,
                width=width,
                height=height,
                title=title,
                line_width=line_width,
                line_color=line_color,
                marker_size=marker_size,
                leaf_data=df_samples,
                leaf_hover_name="sample_id",
                leaf_hover_data=hover_data,
                leaf_color=color_prepped,
                leaf_symbol=symbol_prepped,
                leaf_y=leaf_y,
                leaf_color_discrete_map=color_discrete_map_prepped,
                leaf_category_orders=category_orders_prepped,
                template="simple_white",
                y_axis_title=f"Distance ({distance_metric})",
                y_axis_buffer=0.1,
            )

        # Tidy up.
        fig.update_layout(
            title_font=dict(
                size=title_font_size,
            ),
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)

        # return dict with sample order if for advanced diplotype clustering
        if not return_order_dict and not show:
            return fig
        elif return_order_dict:
            out_dict = {"dist": dist, "samples": gt_samples, "n_snps": n_snps_used}

            order_df = self.extract_dendro_sample_order(fig)
            out_dict["order_data"] = order_df
            return fig, out_dict
        else:
            return None

    def diplotype_pairwise_distances(
        self,
        region: base_params.regions,
        site_mask: base_params.site_mask = DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        site_class: Optional[base_params.site_class] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        distance_metric: dipclust_params.distance_metric = dipclust_params.distance_metric_default,
        random_seed: base_params.random_seed = 42,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "diplotype_pairwise_distances_v1"

        # Normalize params for consistent hash value.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        region_prepped = self._prep_region_cache_param(region=region)
        params = dict(
            region=region_prepped,
            site_mask=site_mask,
            sample_sets=sample_sets_prepped,
            sample_query=sample_query,
            site_class=site_class,
            cohort_size=cohort_size,
            distance_metric=distance_metric,
            random_seed=random_seed,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._diplotype_pairwise_distances(**params)
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results")
        dist: np.ndarray = results["dist"]
        gt_samples: np.ndarray = results["gt_samples"]
        n_snps: int = int(results["n_snps"][()])  # ensure scalar

        return dist, gt_samples, n_snps

    def _diplotype_pairwise_distances(
        self,
        *,
        region,
        site_mask,
        sample_sets,
        sample_query,
        site_class,
        cohort_size,
        distance_metric,
        random_seed,
    ):
        if distance_metric == "cityblock":
            metric = multiallelic_diplotype_mean_cityblock
        elif distance_metric == "euclidean":
            metric = multiallelic_diplotype_mean_sqeuclidean

        # Load haplotypes.
        ds_snps = self.snp_calls(
            region=region,
            sample_query=sample_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        with self._dask_progress(desc="Load genotypes"):
            gt = ds_snps["call_genotype"].data.compute()

        with self._spinner(
            desc="Compute allele counts and remove non-segregating sites"
        ):
            # Compute allele count, remove non-segregating sites.
            ac = allel.GenotypeArray(gt).count_alleles(max_allele=3)
            gt_seg = gt.compress(ac.is_segregating(), axis=0)
            ac_seg = allel.GenotypeArray(gt_seg).to_allele_counts(max_allele=3)
            X = np.ascontiguousarray(np.swapaxes(ac_seg.values, 0, 1))

        # Compute pairwise distances.
        with self._spinner(desc="Compute pairwise distances"):
            dist = multiallelic_diplotype_pdist(X, metric=metric)

        # Extract IDs of samples. Convert to "U" dtype here
        # to allow these to be saved to the results cache.
        gt_samples = ds_snps["sample_id"].values.astype("U")

        return dict(
            dist=dist,
            gt_samples=gt_samples,
            n_snps=np.array(gt_seg.shape[0]),
        )

    def extract_dendro_sample_order(self, fig):
        n_traces = len(fig["data"])
        xs = []
        samples = []
        for i in np.arange(1, n_traces):
            xs.append(fig["data"][i]["x"])
            samples.append(fig["data"][i]["hovertext"])

        return pd.DataFrame(
            {"xs": np.concatenate(xs), "sample_id": np.concatenate(samples)}
        )

    def plotly_track(self, df, colorscale="greys", range_color=None):
        import plotly.express as px

        fig = px.imshow(df, range_color=range_color)
        fig.update_layout(showlegend=False)
        fig.update_traces(dict(showscale=False, coloraxis=None, colorscale=colorscale))
        fig.update_xaxes(visible=False)

        return fig

    @doc(
        summary="Calculate heterozygosity per sample over a region and plot as a track.",
        parameters=dict(
            x_range="The x-axis range of the plot.",
            dendro_sample_id_order="The order of samples in the clustering dendrogram.",
            range_color="The range of the colorscale to use for the plot.",
            color_continuous_scale="The colorscale to use for the plot.",
        ),
    )
    def _plot_dendro_heterozygosity_bar(
        self,
        region: base_params.regions,
        x_range: np.ndarray,
        dendro_sample_id_order: np.ndarray,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        site_mask: base_params.site_mask = DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        range_color=None,
        color_continuous_scale=plotly_params.color_continuous_scale,
    ):
        ds_snps = self.snp_calls(
            region=region,
            sample_query=sample_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            site_mask=site_mask,
            random_seed=random_seed,
        )

        gt = allel.GenotypeDaskArray(ds_snps["call_genotype"].data).compute()
        samples = ds_snps["sample_id"].values.astype("U")

        with self._spinner(desc="Compute heterozygosity"):
            het_per_sample = [
                np.nanmean(allel.heterozygosity_observed(gt[:, [i], :]))
                for i in range(gt.shape[1])
            ]

        df_het = pd.DataFrame(
            {"sample_id": samples, "Sample Heterozygosity": het_per_sample}
        ).set_index("sample_id")

        # order according to dendrogram and transpose, make column names match x_range
        df_het = df_het.loc[dendro_sample_id_order, :].T
        df_het.columns = x_range

        fig = self.plotly_track(
            df=df_het, colorscale=color_continuous_scale, range_color=range_color
        )

        return fig

    def concat_subplots(
        self,
        figures,
        x_range,
        width,
        height,
        row_heights,
        region: base_params.regions,
        n_snps: int,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
    ):
        from plotly.subplots import make_subplots

        title_lines = []
        if sample_sets is not None:
            title_lines.append(f"sample sets: {sample_sets}")
        if sample_query is not None:
            title_lines.append(f"sample query: {sample_query}")
        title_lines.append(f"genomic region: {region} ({n_snps} SNPs)")
        title = "<br>".join(title_lines)

        # make subplots
        fig = make_subplots(
            rows=len(figures),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
        )

        for i, figure in enumerate(figures):
            for trace in range(len(figure["data"])):
                fig.append_trace(figure["data"][trace], row=i + 1, col=1)

        fig.update_xaxes(visible=False)
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            hovermode="closest",
            plot_bgcolor="white",
            xaxis_range=(x_range[0], x_range[-1]),
        )

        return fig

    @doc(
        summary="Perform diplotype clustering with heterozygosity and amino acid variants",
        parameters=dict(
            heterozygosity="Plot heterozygosity track.",
            amino_acids="Plot amino acid variants.",
            leaf_y="Y coordinate at which to plot the leaf markers.",
            filter_min_maf="Filter amino acid variants with alternate allele frequency below this threshold.",
        ),
    )
    def plot_diplotype_clustering_advanced(
        self,
        transcript: base_params.transcript,
        region: base_params.regions,
        site_mask: base_params.site_mask = DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        random_seed: base_params.random_seed = 42,
        cohort_size: Optional[base_params.cohort_size] = None,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        linkage_method: dipclust_params.linkage_method = dipclust_params.linkage_method_default,
        distance_metric: dipclust_params.distance_metric = dipclust_params.distance_metric_default,
        count_sort: Optional[tree_params.count_sort] = None,
        distance_sort: Optional[tree_params.distance_sort] = None,
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        width: plotly_params.width = None,
        height: plotly_params.height = 500,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        leaf_y: int = 0,
        marker_size: plotly_params.marker_size = 5,
        line_width: plotly_params.line_width = 0.5,
        line_color: plotly_params.line_color = "black",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
        heterozygosity: bool = True,
        heterozygosity_colorscale: plotly_params.color_continuous_scale = "Greys",
        amino_acids: bool = True,
        filter_min_maf: float = 0.05,
    ):
        if cohort_size and amino_acids:
            cohort_size = None
            print(
                "Cohort size is not supported with amino acid heatmap. Overriding cohort size to None."
            )

        fig_dendro, out_dict = self.plot_diplotype_clustering(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
            count_sort=count_sort,
            distance_metric=distance_metric,
            cohort_size=cohort_size,
            distance_sort=distance_sort,
            linkage_method=linkage_method,
            color=color,
            symbol=symbol,
            title=title,
            title_font_size=title_font_size,
            width=width,
            height=height,
            show=False,
            renderer=renderer,
            render_mode=render_mode,
            leaf_y=leaf_y,
            marker_size=marker_size,
            line_width=line_width,
            line_color=line_color,
            color_discrete_sequence=color_discrete_sequence,
            color_discrete_map=color_discrete_map,
            category_orders=category_orders,
            legend_sizing=legend_sizing,
            random_seed=random_seed,
            return_order_dict=True,
        )

        x_range = np.sort(out_dict["order_data"]["xs"].to_list())
        n_snps = out_dict["n_snps"]
        dendro_sample_id_order = (
            out_dict["order_data"].sort_values("xs")["sample_id"].to_list()
        )
        frq_sample_id_in_order = ["frq_" + s for s in dendro_sample_id_order]

        figures = []
        row_heights = [0.2]
        figures.append(fig_dendro)

        if heterozygosity:
            fig_het = self._plot_dendro_heterozygosity_bar(
                region=region,
                dendro_sample_id_order=dendro_sample_id_order,
                x_range=x_range,
                sample_sets=sample_sets,
                sample_query=sample_query,
                cohort_size=cohort_size,
                site_mask=site_mask,
                color_continuous_scale=heterozygosity_colorscale,
                random_seed=random_seed,
            )
            figures.append(fig_het)
            row_heights.append(0.012)

        if transcript and amino_acids:
            # load genotypes at amino acid variants for each sample
            df_snps = self.aa_individual_allele_frequencies(
                transcript=transcript,
                sample_query=sample_query,
                sample_sets=sample_sets,
                site_mask=site_mask,
            )

            # set to diplotype cluster order
            df_snps = df_snps.reset_index(drop=True).set_index("label")
            df_snps = (
                df_snps.filter(like="frq").fillna(0).loc[:, frq_sample_id_in_order]
            )
            df_snps.columns = x_range

            if filter_min_maf:
                df_snps = df_snps.assign(af=lambda x: x.sum(axis=1) / x.shape[1])
                df_snps = df_snps.query("af > @filter_min_maf").drop(columns="af")

            # aa_height = np.max([df_snps.shape[0]/100, 0.1]) # minimum height of 0.1
            fig_aa = self.plotly_track(df_snps)
            figures.append(fig_aa)
            row_heights.append(0.2)

        fig = self.concat_subplots(
            figures=figures,
            x_range=x_range,
            width=width,
            height=height,
            row_heights=row_heights,
            sample_sets=sample_sets,
            sample_query=sample_query,
            region=region,
            n_snps=n_snps,
        )

        if transcript and amino_acids:
            # add lines to aa plot
            aa_idx = len(figures)
            fig.add_hline(y=-0.5, line_width=1, line_color="grey", row=aa_idx, col=1)
            for i, y in enumerate(df_snps.index.to_list()):
                fig.add_hline(
                    y=i + 0.5, line_width=1, line_color="grey", row=aa_idx, col=1
                )

            fig["layout"][f"yaxis{aa_idx}"]["title"] = f"{transcript} amino acids"
            fig.update_xaxes(
                showline=True,
                linecolor="grey",
                linewidth=1,
                row=aa_idx,
                col=1,
                mirror=True,
            )
            fig.update_yaxes(
                showline=True,
                linecolor="grey",
                linewidth=1,
                row=aa_idx,
                col=1,
                mirror=True,
            )

        fig["layout"]["yaxis"]["title"] = "Distance (manhattan)"

        if show:
            fig.show(renderer=renderer)
            return None
        else:
            return fig
