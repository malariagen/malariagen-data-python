from typing import Optional, Tuple

import allel  # type: ignore
import numpy as np
import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, check_types, pdist_abs_hamming
from ..plotly_dendrogram import plot_dendrogram
from . import (
    base_params,
    plotly_params,
    tree_params,
    hap_params,
    clustering_params,
    hapclust_params,
)
from .snp_data import AnophelesSnpData
from .snp_frq import AA_CHANGE_QUERY
from .hap_data import AnophelesHapData


class AnophelesHapClustAnalysis(AnophelesHapData, AnophelesSnpData):
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
        summary="""
            Hierarchically cluster haplotypes in region and produce an interactive plot.
        """,
    )
    def plot_haplotype_clustering(
        self,
        region: base_params.regions,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        linkage_method: hapclust_params.linkage_method = hapclust_params.linkage_method_default,
        count_sort: Optional[tree_params.count_sort] = None,
        distance_sort: Optional[tree_params.distance_sort] = None,
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        width: plotly_params.fig_width = None,
        height: plotly_params.fig_height = 500,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        leaf_y: clustering_params.leaf_y = 0,
        marker_size: plotly_params.marker_size = 5,
        line_width: plotly_params.line_width = 0.5,
        line_color: plotly_params.line_color = "black",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> plotly_params.figure:
        import sys

        # Normalise params.
        if count_sort is None and distance_sort is None:
            count_sort = True
            distance_sort = False

        # This is needed to avoid RecursionError on some haplotype clustering analyses
        # with larger numbers of haplotypes.
        sys.setrecursionlimit(10_000)

        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )

        # Compute pairwise distances.
        dist, phased_samples, n_snps_used = self.haplotype_pairwise_distances(
            region=region,
            analysis=analysis,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Align sample metadata with haplotypes.
        df_samples_phased = (
            df_samples.set_index("sample_id").loc[phased_samples.tolist()].reset_index()
        )

        # Normalise color and symbol parameters.
        symbol_prepped = self._setup_sample_symbol(
            data=df_samples_phased,
            symbol=symbol,
        )
        del symbol
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=df_samples_phased,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence

        # Repeat the dataframe so there is one row of metadata for each haplotype.
        df_haps = pd.DataFrame(np.repeat(df_samples_phased.values, 2, axis=0))
        df_haps.columns = df_samples_phased.columns
        leaf_data = df_haps.assign(sample_id=_make_unique(df_haps.sample_id))

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
            fig, leaf_data = plot_dendrogram(
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
                leaf_data=leaf_data,
                leaf_hover_name="sample_id",
                leaf_hover_data=hover_data,
                leaf_color=color_prepped,
                leaf_symbol=symbol_prepped,
                leaf_y=leaf_y,
                leaf_color_discrete_map=color_discrete_map_prepped,
                leaf_category_orders=category_orders_prepped,
                template="simple_white",
                y_axis_title="Distance (no. SNPs)",
                y_axis_buffer=1,
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
            return None
        else:
            return {
                "figure": fig,
                "n_snps": n_snps_used,
                "dist": dist,
                "dist_samples": phased_samples,
                "leaf_data":leaf_data,
            }

    @doc(
        summary="""
            Compute pairwise distances between haplotypes.
        """,
        returns=dict(
            dist="Pairwise distance.",
            phased_samples="Sample identifiers for haplotypes.",
            n_snps="Number of SNPs used.",
        ),
    )
    def haplotype_pairwise_distances(
        self,
        region: base_params.regions,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "haplotype_pairwise_distances"

        # Normalize params for consistent hash value.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        region_prepped = self._prep_region_cache_param(region=region)
        params = dict(
            region=region_prepped,
            analysis=analysis,
            sample_sets=sample_sets_prepped,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._haplotype_pairwise_distances(
                chunks=chunks, inline_array=inline_array, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results")
        dist: np.ndarray = results["dist"]
        phased_samples: np.ndarray = results["phased_samples"]
        n_snps: int = int(results["n_snps"][()])  # ensure scalar

        return dist, phased_samples, n_snps

    def _haplotype_pairwise_distances(
        self,
        *,
        region,
        analysis,
        sample_sets,
        sample_query,
        sample_query_options,
        cohort_size,
        random_seed,
        chunks,
        inline_array,
    ):
        from scipy.spatial.distance import squareform  # type: ignore

        # Load haplotypes.
        ds_haps = self.haplotypes(
            region=region,
            analysis=analysis,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute().values

        # Compute allele count, remove non-segregating sites.
        ac = allel.HaplotypeArray(ht).count_alleles(max_allele=1)
        ht_seg = ht[ac.is_segregating()]

        # Transpose memory layout for faster hamming distance calculations.
        ht_t = np.ascontiguousarray(ht_seg.T)

        # Compute pairwise distances.
        with self._spinner(desc="Compute pairwise distances"):
            dist_sq = pdist_abs_hamming(ht_t)
        dist = squareform(dist_sq)

        # Extract IDs of phased samples. Convert to "U" dtype here
        # to allow these to be saved to the results cache.
        phased_samples = ds_haps["sample_id"].values.astype("U")

        return dict(
            dist=dist,
            phased_samples=phased_samples,
            n_snps=np.array(ht.shape[0]),
        )

    def plot_haplotype_clustering_advanced(
        self,
        region,
        analysis,
        snp_transcript=None,
        snp_colorscale="Greys",
        snp_filter_min_maf=0.05,
        snp_query=AA_CHANGE_QUERY,
        sample_sets=None,
        sample_query=None,
        sample_query_options=None,
        random_seed=42,
        cohort_size=None,
        cut_height=None,
        min_cluster_size=None,
        color=None,
        symbol=None,
        linkage_method="complete",
        count_sort=None,
        distance_sort=None,
        title=True,
        title_font_size=14,
        width=None,
        dendrogram_height=300,
        snp_row_height=25,
        show=True,
        renderer=None,
        render_mode="svg",
        leaf_y=0,
        marker_size=5,
        line_width=0.5,
        line_color="black",
        color_discrete_sequence=None,
        color_discrete_map=None,
        category_orders=None,
        legend_sizing="constant",
        chunks="native",
        inline_array=True,
    ):
        import plotly.express as px
        import plotly.graph_objects as go

        if cohort_size and snp_transcript:
            cohort_size = None
            print(
                "Cohort size is not supported with amino acid heatmap. Overriding cohort size to None."
            )

        res = self.plot_haplotype_clustering(
            region=region,
            analysis=analysis,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            count_sort=count_sort,
            cohort_size=cohort_size,
            distance_sort=distance_sort,
            linkage_method=linkage_method,
            color=color,
            symbol=symbol,
            title=title,
            title_font_size=title_font_size,
            width=width,
            height=dendrogram_height,
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
            chunks=chunks,
            inline_array=inline_array,
        )

        fig_dendro = res["figure"]
        n_snps_cluster = res["n_snps"]
        leaf_data = res["leaf_data"]
        dendro_sample_id_order = np.asarray(leaf_data["sample_id"].to_list())

        figures = [fig_dendro]
        subplot_heights = [dendrogram_height]
        
        if cut_height and min_cluster_size:
            cluster_col_list = px.colors.sequential.Turbo.copy()
            cluster_col_list.insert(0, 'white')
            
            df_clusters = self.cut_dist_tree(
                dist=res['dist'], 
                dist_samples=_make_unique(np.repeat(res['dist_samples'], 2)),
                dendro_sample_id_order=dendro_sample_id_order,
                linkage_method=linkage_method, 
                cut_height=cut_height,
                min_cluster_size=min_cluster_size, 
                count_sort=count_sort, 
                distance_sort=distance_sort
            )

            leaf_data = leaf_data.merge(df_clusters.T.reset_index())
            
            snp_trace = go.Heatmap(
                z=df_clusters.values,
                y=df_clusters.index.to_list(),
                colorscale=cluster_col_list,
                showlegend=False,
                showscale=False,
            )
            figures.append(snp_trace)
            subplot_heights.append(25)
            
            
        figures, subplot_heights, df_haps = self._insert_hapclust_snp_trace(
            transcript=snp_transcript,
            snp_query=snp_query,
            figures=figures,
            subplot_heights=subplot_heights,
            sample_sets=sample_sets,
            sample_query=sample_query,
            analysis=analysis,
            dendro_sample_id_order=dendro_sample_id_order,
            snp_filter_min_maf=snp_filter_min_maf,
            snp_colorscale=snp_colorscale,
            snp_row_height=snp_row_height,
            chunks=chunks,
            inline_array=inline_array,
        )
        n_aa = df_haps.shape[0]

        # Calculate total height based on subplot heights, plus a fixed
        # additional component to allow for title, axes etc.
        height = sum(subplot_heights) + 50
        fig = self._dipclust_concat_subplots(
            figures=figures,
            width=width,
            height=height,
            row_heights=subplot_heights,
            sample_sets=sample_sets,
            sample_query=sample_query,  # Only uses query for title.
            region=region,
            n_snps=n_snps_cluster,
        )

        fig["layout"]["yaxis"]["title"] = f"Distance"
        fig.update_layout(
            title_font=dict(
                size=title_font_size,
            ),
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )
        
        if snp_transcript:
        # add lines to aa plot
            aa_idx = len(figures)
            fig.add_hline(y=-0.5, line_width=1, line_color="grey", row=aa_idx, col=1)
            for i in range(n_aa):
                fig.add_hline(y=i+0.5, line_width=1, line_color="grey", row=aa_idx, col=1)
            
            fig.update_xaxes(showline = True, linecolor = 'grey', linewidth = 1, row = aa_idx, col = 1, mirror = True)
            fig.update_yaxes(showline = True, linecolor = 'grey', linewidth = 1, row = aa_idx, col = 1, mirror = True)

        if show:
            fig.show(renderer=renderer)
            return None
        else:
            return fig, leaf_data, df_haps
        
        
    def transcript_haplotypes(
        self,
        transcript,
        sample_sets,
        sample_query,
        analysis,
        snp_query,
        chunks,
        inline_array
    ):
        """
        Extract haplotype calls for a given transcript.
        """
        
        # Get SNP genotype allele counts for the transcript, applying snp_query
        df_snps = self.snp_genotype_allele_counts(
            transcript=transcript,
            sample_query=sample_query,
            sample_sets=sample_sets,
            site_mask=None,
            snp_query=snp_query,
            chunks=chunks,
            inline_array=inline_array
        )

        # Add a unique variant identifier: "position-alt_allele"
        df_snps = df_snps.assign(
            pos_alt=lambda x: x.position.astype(str) + "-" + x.alt_allele
        )

        # Get haplotypes for the transcript
        ds_haps = self.haplotypes(
            region=transcript,
            sample_sets=sample_sets,
            sample_query=sample_query,
            analysis=analysis,
            chunks=chunks,
            inline_array=inline_array
        )

        # Convert genotype calls to haplotypes
        haps = allel.GenotypeArray(ds_haps["call_genotype"].values).to_haplotypes()
        h_pos = allel.SortedIndex(ds_haps["variant_position"].values)
        h_alts = ds_haps["variant_allele"].values.astype(str)[:, 1]
        h_pos_alts = np.array([f"{pos}-{h_alts[i]}" for i, pos in enumerate(h_pos)])
        sample_ids = ds_haps["sample_id"].values

        # Filter haplotypes to SNPs present in df_snps
        df_snps = df_snps.query("pos_alt in @h_pos_alts")
        label = df_snps.label.values
        haps_bool = np.isin(h_pos_alts, df_snps.pos_alt)
        haps = haps.compress(haps_bool)

        # Build haplotype DataFrame
        df_haps = pd.DataFrame(
            haps,
            index=label,
            columns=_make_unique(np.repeat(sample_ids, 2))  # two haplotypes per sample
        )

        return df_haps

    def _insert_hapclust_snp_trace(
        self,
        figures,
        subplot_heights,
        transcript,
        analysis,
        dendro_sample_id_order: np.ndarray,
        snp_filter_min_maf: float,
        snp_colorscale,
        snp_query,
        snp_row_height,
        sample_sets,
        sample_query,
        chunks,
        inline_array,
    ):
        from plotly import graph_objects as go
        
        # load genotype allele counts at SNP variants for each sample
        df_haps = self.transcript_haplotypes(
            transcript=transcript,
            snp_query=snp_query,
            sample_query=sample_query,
            sample_sets=sample_sets,
            analysis=analysis,
            chunks=chunks,
            inline_array=inline_array
        )
        
        # set to diplotype cluster order
        df_haps = df_haps.loc[
            :, dendro_sample_id_order
        ]

        if snp_filter_min_maf:
            df_haps = df_haps.assign(af=lambda x: x.sum(axis=1) / x.shape[1])
            df_haps = df_haps.query("af > @snp_filter_min_maf").drop(columns="af")

        if not df_haps.empty:
            snp_trace = go.Heatmap(
                z=df_haps.values,
                y=df_haps.index.to_list(),
                colorscale=snp_colorscale,
                showlegend=False,
                showscale=False,
            )
        else:
            snp_trace = None
            
        if snp_trace:
            figures.append(snp_trace)
            subplot_heights.append(snp_row_height * df_haps.shape[0])
        else:
            print(
                f"No SNPs were found below {snp_filter_min_maf} allele frequency. Omitting SNP genotype plot."
            )
        return figures, subplot_heights, df_haps


    def cut_dist_tree(self, dist, dist_samples, dendro_sample_id_order, linkage_method, cut_height, min_cluster_size, count_sort, distance_sort):
        """
        Create a one-row DataFrame with haplotype_ids as columns and cluster assignments as values

        Parameters:
        -----------
        dist : ndarray
            distance array
        dist_samples : array-like
            List/array of individual identifiers (haplotype_ids)
        linkage_method : str
            Method used to calculate the linkage matrix
        cut_height : float
            Height at which to cut the dendrogram
        min_cluster_size : int, default=1
            Minimum number of individuals required in a cluster to be included
        count_sort : bool
            Whether to sort clusters by count
        distance_sort : bool
            Whether to sort clusters by distance

        Returns:
        --------
        pd.DataFrame
            One-row DataFrame with haplotype_ids as columns and assigned cluster numbers (1...n) as values
        """
        from scipy.cluster.hierarchy import linkage, fcluster

        Z = linkage(dist, method=linkage_method)

        # Get cluster assignments for each individual
        cluster_assignments = fcluster(
            Z, cut_height, criterion='distance'
        )
        
        # Create initial DataFrame
        df = pd.DataFrame({
            'sample_id': dist_samples,
            'cluster_id': _filter_and_remap(cluster_assignments, x=min_cluster_size)
        }).set_index('sample_id').T.loc[:, dendro_sample_id_order]

        return df

def _filter_and_remap(arr, x):
    from collections import Counter
    # Get unique values that appear >= x times
    valid_values = [val for val, count in Counter(arr).items() if count >= x]
    # Create mapping to 1, 2, 3, ..., n
    mapping = {val: i + 1 for i, val in enumerate(sorted(valid_values))}
    # Apply transformation
    return np.array([mapping.get(val, 0) for val in arr])


def _make_unique(values):
    value_counts = {}
    unique_values = []
    
    for value in values:
        if value in value_counts:
            value_counts[value] += 1
            unique_values.append(f"{value}_{value_counts[value]}")
        else:
            value_counts[value] = 0
            unique_values.append(f"{value}_{value_counts[value]}")
    
    return np.array(unique_values)