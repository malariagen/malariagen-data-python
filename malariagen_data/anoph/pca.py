from typing import Optional, Tuple

import allel  # type: ignore
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, check_types, jitter
from . import base_params, pca_params, plotly_params
from .snp_data import AnophelesSnpData


class AnophelesPca(
    AnophelesSnpData,
):
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
            Run a principal components analysis (PCA) using biallelic SNPs from
            the selected genome region and samples.
        """,
        extended_summary="""
            .. versionchanged:: 8.0.0
               SNP ascertainment has changed slightly.

            This function uses biallelic SNPs as input to the PCA. The ascertainment
            of SNPs to include has changed slightly in version 8.0.0 and therefore
            the results of this function may also be slightly different. Previously,
            SNPs were required to be biallelic and one of the observed alleles was
            required to be the reference allele. Now SNPs just have to be biallelic.

            The following additional parameters were also added in version 8.0.0:
            `site_class`, `cohort_size`, `min_cohort_size`, `max_cohort_size`,
            `random_seed`.

        """,
        returns=("df_pca", "evr"),
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Results of this computation will be cached and re-used if
            the `results_cache` parameter was set when instantiating the API client.
        """,
    )
    def pca(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        n_components: pca_params.n_components = pca_params.n_components_default,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        min_minor_ac: Optional[
            base_params.min_minor_ac
        ] = pca_params.min_minor_ac_default,
        max_missing_an: Optional[
            base_params.max_missing_an
        ] = pca_params.max_missing_an_default,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> Tuple[pca_params.df_pca, pca_params.evr]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "pca_v3"

        # Normalize params for consistent hash value.
        (
            sample_sets_prepped,
            sample_indices_prepped,
        ) = self._prep_sample_selection_cache_params(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
        )
        region_prepped = self._prep_region_cache_param(region=region)
        site_mask_prepped = self._prep_optional_site_mask_param(site_mask=site_mask)
        params = dict(
            region=region_prepped,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets_prepped,
            sample_indices=sample_indices_prepped,
            site_mask=site_mask_prepped,
            site_class=site_class,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_components=n_components,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._pca(chunks=chunks, inline_array=inline_array, **params)
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results.
        coords = results["coords"]
        evr = results["evr"]
        samples = results["samples"]

        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_indices=sample_indices_prepped,
        )

        # Ensure aligned with genotype data.
        df_samples = df_samples.set_index("sample_id").loc[samples].reset_index()

        # Combine coords and sample metadata.
        df_coords = pd.DataFrame(
            {f"PC{i + 1}": coords[:, i] for i in range(coords.shape[1])}
        )
        df_pca = df_samples.join(df_coords, how="inner")

        return df_pca, evr

    def _pca(
        self,
        *,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_indices,
        site_mask,
        site_class,
        min_minor_ac,
        max_missing_an,
        n_components,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        chunks,
        inline_array,
    ):
        # Load diplotypes.
        gn, samples = self.biallelic_diplotypes(
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            site_class=site_class,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        with self._spinner(desc="Compute PCA"):
            # Remove any sites where all genotypes are identical.
            loc_var = np.any(gn != gn[:, 0, np.newaxis], axis=1)
            gn_var = np.compress(loc_var, gn, axis=0)

            # Run the PCA.
            coords, model = allel.pca(gn_var, n_components=n_components)

            # Work around sign indeterminacy.
            for i in range(coords.shape[1]):
                c = coords[:, i]
                if np.abs(c.min()) > np.abs(c.max()):
                    coords[:, i] = c * -1

        results = dict(
            samples=samples, coords=coords, evr=model.explained_variance_ratio_
        )
        return results

    @check_types
    @doc(
        summary="""
            Plot explained variance ratios from a principal components analysis
            (PCA) using a plotly bar plot.
        """,
        parameters=dict(
            kwargs="Passed through to px.bar().",
        ),
    )
    def plot_pca_variance(
        self,
        evr: pca_params.evr,
        width: plotly_params.fig_width = 900,
        height: plotly_params.fig_height = 400,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        # Prepare plotting variables.
        y = evr * 100  # convert to percent
        x = [str(i + 1) for i in range(len(y))]

        # Set up plotting options.
        plot_kwargs = dict(
            labels={
                "x": "Principal component",
                "y": "Explained variance (%)",
            },
            template="simple_white",
            width=width,
            height=height,
        )
        # Apply any user overrides.
        plot_kwargs.update(kwargs)

        # Make a bar plot.
        fig = px.bar(x=x, y=y, **plot_kwargs)

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Plot sample coordinates from a principal components analysis (PCA)
            as a plotly scatter plot.
        """,
        parameters=dict(
            opacity="Marker opacity.",
            kwargs="Passed through to `px.scatter()`",
        ),
    )
    def plot_pca_coords(
        self,
        data: pca_params.df_pca,
        x: plotly_params.x = "PC1",
        y: plotly_params.y = "PC2",
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        opacity: float = 0.9,
        jitter_frac: plotly_params.jitter_frac = 0.02,
        random_seed: base_params.random_seed = 42,
        width: plotly_params.fig_width = 900,
        height: plotly_params.fig_height = 600,
        marker_size: plotly_params.marker_size = 10,
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        **kwargs,
    ) -> plotly_params.figure:
        # Copy input data to avoid overwriting.
        data = data.copy()

        # Apply jitter if desired - helps spread out points when tightly clustered.
        if jitter_frac:
            np.random.seed(random_seed)
            data[x] = jitter(data[x], jitter_frac)
            data[y] = jitter(data[y], jitter_frac)

        # Convenience variables.
        data["country_location"] = data["country"] + " - " + data["location"]

        # Normalise color and symbol parameters.
        symbol_prepped = self._setup_sample_symbol(
            data=data,
            symbol=symbol,
        )
        del symbol
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=data,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence
        del category_orders

        # Configure hover data.
        hover_data = self._setup_sample_hover_data_plotly(
            color=color_prepped, symbol=symbol_prepped
        )

        # Set up plotting options.
        plot_kwargs = dict(
            width=width,
            height=height,
            color=color_prepped,
            symbol=symbol_prepped,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
            template="simple_white",
            hover_name="sample_id",
            hover_data=hover_data,
            opacity=opacity,
            render_mode=render_mode,
        )

        # Apply any user overrides.
        plot_kwargs.update(kwargs)

        # 2D scatter plot.
        fig = px.scatter(data, x=x, y=y, **plot_kwargs)

        # Tidy up.
        fig.update_layout(
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )
        fig.update_traces(marker={"size": marker_size})

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Plot sample coordinates from a principal components analysis (PCA)
            as a plotly 3D scatter plot.
        """,
        parameters=dict(
            kwargs="Passed through to `px.scatter_3d()`",
        ),
    )
    def plot_pca_coords_3d(
        self,
        data: pca_params.df_pca,
        x: plotly_params.x = "PC1",
        y: plotly_params.y = "PC2",
        z: plotly_params.z = "PC3",
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        jitter_frac: plotly_params.jitter_frac = 0.02,
        random_seed: base_params.random_seed = 42,
        width: plotly_params.fig_width = 900,
        height: plotly_params.fig_height = 600,
        marker_size: plotly_params.marker_size = 5,
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        # Copy input data to avoid overwriting.
        data = data.copy()

        # Apply jitter if desired - helps spread out points when tightly clustered.
        if jitter_frac:
            np.random.seed(random_seed)
            data[x] = jitter(data[x], jitter_frac)
            data[y] = jitter(data[y], jitter_frac)
            data[z] = jitter(data[z], jitter_frac)

        # Convenience variables.
        data["country_location"] = data["country"] + " - " + data["location"]

        # Normalise color and symbol parameters.
        symbol_prepped = self._setup_sample_symbol(
            data=data,
            symbol=symbol,
        )
        del symbol
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=data,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence
        del category_orders

        # Configure hover data.
        hover_data = self._setup_sample_hover_data_plotly(
            color=color_prepped, symbol=symbol_prepped
        )

        # Set up plotting options.
        plot_kwargs = dict(
            width=width,
            height=height,
            hover_name="sample_id",
            hover_data=hover_data,
            color=color_prepped,
            symbol=symbol_prepped,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
        )

        # Apply any user overrides.
        plot_kwargs.update(kwargs)

        # 3D scatter plot.
        fig = px.scatter_3d(data, x=x, y=y, z=z, **plot_kwargs)

        # Tidy up.
        fig.update_layout(
            scene=dict(aspectmode="cube"),
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )
        fig.update_traces(marker={"size": marker_size})

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig
