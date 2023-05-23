from textwrap import dedent
from typing import Optional

import numpy as np
import xarray as xr
from numpydoc_decorator import doc

from ..util import DIM_SAMPLE, check_types, init_zarr_store, simple_xarray_concat
from . import aim_params, base_params
from .genome_features import AnophelesGenomeFeaturesData
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata


class AnophelesHapData(
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

        # Set up caches.
        self._cache_aim_variants = dict()

    @check_types
    @doc(
        summary="Access ancestry informative marker variants.",
        returns="A dataset containing AIM positions and discriminating alleles.",
    )
    def aim_variants(self, aims: aim_params.aims) -> xr.Dataset:
        self._require_aim_analysis()
        try:
            ds = self._cache_aim_variants[aims]
        except KeyError:
            analysis = self._aim_analysis
            path = f"{self._base_path}/reference/aim_defs_{analysis}/{aims}.zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            ds = xr.open_zarr(store, concat_characters=False)
            ds = ds.set_coords(["variant_contig", "variant_position"])
            self._cache_aim_variants[aims] = ds
        return ds.copy(deep=False)

    def _aim_calls_dataset(self, *, aims, sample_set):
        self._require_aim_analysis()
        release = self.lookup_release(sample_set=sample_set)
        release_path = self._release_to_path(release)
        analysis = self._aim_analysis
        path = f"{self._base_path}/{release_path}/aim_calls_{analysis}/{sample_set}/{aims}.zarr"
        store = init_zarr_store(fs=self._fs, path=path)
        ds = xr.open_zarr(store=store, concat_characters=False)
        ds = ds.set_coords(["variant_contig", "variant_position", "sample_id"])
        return ds

    # TODO add type annotations and @doc
    def aim_calls(
        self,
        aims,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
    ):
        """Access ancestry informative marker SNP sites, alleles and genotype
        calls.

        Parameters
        ----------
        aims : {'gamb_vs_colu', 'gambcolu_vs_arab'}
            Which ancestry informative markers to use.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".

        Returns
        -------
        ds : xarray.Dataset
            A dataset containing AIM SNP sites, alleles and genotype calls.

        """
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)

        debug("access SNP calls and concatenate multiple sample sets and/or regions")
        ly = []
        for s in sample_sets:
            y = self._aim_calls_dataset(
                aims=aims,
                sample_set=s,
            )
            ly.append(y)

        debug("concatenate data from multiple sample sets")
        ds = simple_xarray_concat(ly, dim=DIM_SAMPLE)

        debug("handle sample query")
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)

        return ds

    # TODO add type annotations and @doc
    # TODO generalise colors
    def plot_aim_heatmap(
        self,
        aims,
        sample_sets=None,
        sample_query=None,
        sort=True,
        row_height=4,
        colors="T10",
        xgap=0,
        ygap=0.5,
        show=True,
        renderer=None,
    ):
        """Plot a heatmap of ancestry-informative marker (AIM) genotypes.

        Parameters
        ----------
        aims : {'gamb_vs_colu', 'gambcolu_vs_arab'}
            Which ancestry informative markers to use.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        sort : bool, optional
            If true (default), sort the samples by the total fraction of AIM
            alleles for the second species in the comparison.
        row_height : int, optional
            Height per sample in px.
        colors : str, optional
            Choose your favourite color palette.
        xgap : float, optional
            Creates lines between columns (variants).
        ygap : float, optional
            Creates lines between rows (samples).

        Returns
        -------
        fig : plotly.graph_objects.Figure

        """

        debug = self._log.debug

        import allel
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        debug("load AIM calls")
        ds = self.aim_calls(
            aims=aims,
            sample_sets=sample_sets,
            sample_query=sample_query,
        ).compute()
        samples = ds["sample_id"].values
        variant_contig = ds["variant_contig"].values

        debug("count variants per contig")
        contigs = ds.attrs["contigs"]
        col_widths = [
            np.count_nonzero(variant_contig == contigs.index(contig))
            for contig in contigs
        ]
        debug(col_widths)

        debug("access and transform genotypes")
        gt = allel.GenotypeArray(ds["call_genotype"].values)
        gn = gt.to_n_alt(fill=-1)

        if sort:
            debug("sort by AIM fraction")
            ac = np.sum(gt == 1, axis=(0, 2))
            an = np.sum(gt >= 0, axis=(0, 2))
            af = ac / an
            ix_sorted = np.argsort(af)
            gn = np.take(gn, ix_sorted, axis=1)
            samples = np.take(samples, ix_sorted, axis=0)

        debug("set up colors")
        # https://en.wiktionary.org/wiki/abandon_hope_all_ye_who_enter_here
        if colors.lower() == "plotly":
            palette = px.colors.qualitative.Plotly
            color_gc = palette[3]
            color_gc_a = palette[9]
            color_a = palette[2]
            color_g = palette[0]
            color_g_c = palette[9]
            color_c = palette[1]
            color_m = "white"
        elif colors.lower() == "set1":
            palette = px.colors.qualitative.Set1
            color_gc = palette[3]
            color_gc_a = palette[4]
            color_a = palette[2]
            color_g = palette[1]
            color_g_c = palette[5]
            color_c = palette[0]
            color_m = "white"
        elif colors.lower() == "g10":
            palette = px.colors.qualitative.G10
            color_gc = palette[4]
            color_gc_a = palette[2]
            color_a = palette[3]
            color_g = palette[0]
            color_g_c = palette[2]
            color_c = palette[8]
            color_m = "white"
        elif colors.lower() == "t10":
            palette = px.colors.qualitative.T10
            color_gc = palette[6]
            color_gc_a = palette[5]
            color_a = palette[4]
            color_g = palette[0]
            color_g_c = palette[5]
            color_c = palette[2]
            color_m = "white"
        else:
            raise ValueError("unsupported colors")
        if aims == "gambcolu_vs_arab":
            colors = [color_m, color_gc, color_gc_a, color_a]
        else:
            colors = [color_m, color_g, color_g_c, color_c]
        species = aims.split("_vs_")

        debug("create subplots")
        fig = make_subplots(
            rows=1,
            cols=len(contigs),
            shared_yaxes=True,
            column_titles=contigs,
            row_titles=None,
            column_widths=col_widths,
            x_title="Variants",
            y_title="Samples",
            horizontal_spacing=0.01,
            vertical_spacing=0.01,
        )

        for j, contig in enumerate(contigs):
            debug(f"plot {contig}")
            loc_contig = variant_contig == j
            gn_contig = np.compress(loc_contig, gn, axis=0)
            fig.add_trace(
                go.Heatmap(
                    y=samples,
                    z=gn_contig.T,
                    # construct a discrete color scale
                    # https://plotly.com/python/colorscales/#constructing-a-discrete-or-discontinuous-color-scale
                    colorscale=[
                        [0 / 4, colors[0]],
                        [1 / 4, colors[0]],
                        [1 / 4, colors[1]],
                        [2 / 4, colors[1]],
                        [2 / 4, colors[2]],
                        [3 / 4, colors[2]],
                        [3 / 4, colors[3]],
                        [4 / 4, colors[3]],
                    ],
                    zmin=-1.5,
                    zmax=2.5,
                    xgap=xgap,
                    ygap=ygap,  # this creates faint lines between rows
                    colorbar=dict(
                        title="AIM genotype",
                        tickmode="array",
                        tickvals=[-1, 0, 1, 2],
                        ticktext=[
                            "missing",
                            f"{species[0]}/{species[0]}",
                            f"{species[0]}/{species[1]}",
                            f"{species[1]}/{species[1]}",
                        ],
                        len=100,
                        lenmode="pixels",
                        y=1,
                        yanchor="top",
                        outlinewidth=1,
                        outlinecolor="black",
                    ),
                    hovertemplate=dedent(
                        """
                        Variant index: %{x}<br>
                        Sample: %{y}<br>
                        Genotype: %{z}
                        <extra></extra>
                    """
                    ),
                ),
                row=1,
                col=j + 1,
            )

        fig.update_xaxes(
            tickmode="array",
            tickvals=[],
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=[],
        )

        fig.update_layout(
            title=f"AIMs - {aims}",
            height=max(300, row_height * len(samples) + 100),
        )

        if show:
            fig.show(renderer=renderer)
            return None
        else:
            return fig
