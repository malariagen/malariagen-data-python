from textwrap import dedent
from typing import Dict, Optional

import allel
import numpy as np
import plotly.graph_objects as go
import xarray as xr
from numpydoc_decorator import doc
from plotly.subplots import make_subplots as go_make_subplots

from malariagen_data.anoph import plotly_params

from ..util import DIM_SAMPLE, check_types, init_zarr_store, simple_xarray_concat
from . import aim_params, base_params
from .genome_features import AnophelesGenomeFeaturesData
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata

# Note that the AIM variants and genotype calls data are stored using
# the xarray zarr format, and so it is possible to load directly
# into an xarray dataset using the xr.open_zarr() function.
#
# For more information see also:
#
# https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html
# https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html


class AnophelesAimData(
    AnophelesSampleMetadata, AnophelesGenomeFeaturesData, AnophelesGenomeSequenceData
):
    def __init__(
        self,
        aim_ids: Optional[aim_params.aim_ids] = None,
        aim_palettes: Optional[aim_params.aim_palettes] = None,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # Store possible values for the `aims` parameter.
        # TODO Consider moving this to data resource configuration.
        self._aim_ids = aim_ids
        self._aim_palettes = aim_palettes

        # Set up caches.
        self._cache_aim_variants: Dict[str, xr.Dataset] = dict()

    @property
    def aim_ids(self) -> aim_params.aim_ids:
        if self._aim_ids is not None:
            return tuple(self._aim_ids)
        else:
            return ()

    def _prep_aims_param(self, *, aims: aim_params.aims) -> str:
        aims = aims.lower()
        if aims in self.aim_ids:
            return aims
        else:
            raise ValueError(f"Invalid aims parameter, must be one of {self.aim_ids}.")

    @check_types
    @doc(
        summary="Access ancestry informative marker variants.",
        returns="A dataset containing AIM positions and discriminating alleles.",
    )
    def aim_variants(self, aims: aim_params.aims) -> xr.Dataset:
        self._require_aim_analysis()
        aims = self._prep_aims_param(aims=aims)
        try:
            ds = self._cache_aim_variants[aims]
        except KeyError:
            # Determine which AIM analysis to load.
            analysis = self._aim_analysis

            # Build the path to the zarr data.
            path = f"{self._base_path}/reference/aim_defs_{analysis}/{aims}.zarr"

            # Initialise and open the zarr data.
            store = init_zarr_store(fs=self._fs, path=path)
            ds = xr.open_zarr(store, concat_characters=False)
            ds = ds.set_coords(["variant_contig", "variant_position"])

            # Cache to avoid re-opening which saves a little time for the user.
            self._cache_aim_variants[aims] = ds

        # N.B., return a copy so any modifications to the dataset made by the
        # user to not affect the cached dataset.
        return ds.copy(deep=False)

    def _aim_calls_dataset(self, *, aims, sample_set):
        self._require_aim_analysis()

        # Build the path to the zarr data.
        release = self.lookup_release(sample_set=sample_set)
        release_path = self._release_to_path(release)
        analysis = self._aim_analysis
        path = f"{self._base_path}/{release_path}/aim_calls_{analysis}/{sample_set}/{aims}.zarr"

        # Initialise and open the zarr data.
        store = init_zarr_store(fs=self._fs, path=path)
        ds = xr.open_zarr(store=store, concat_characters=False)
        ds = ds.set_coords(["variant_contig", "variant_position", "sample_id"])
        return ds

    @check_types
    @doc(
        summary="""
            Access ancestry informative marker SNP sites, alleles and genotype
            calls.
        """,
        returns="""
            A dataset containing AIM SNP sites, alleles and genotype calls.
        """,
    )
    def aim_calls(
        self,
        aims: aim_params.aims,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
    ) -> xr.Dataset:
        self._require_aim_analysis()

        # Normalise parameters.
        aims = self._prep_aims_param(aims=aims)
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        del sample_sets

        # Access SNP calls and concatenate multiple sample sets and/or regions.
        ly = []
        for s in sample_sets_prepped:
            y = self._aim_calls_dataset(
                aims=aims,
                sample_set=s,
            )
            ly.append(y)

        # Concatenate data from multiple sample sets.
        ds = simple_xarray_concat(ly, dim=DIM_SAMPLE)

        # Handle sample query.
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets_prepped)
            loc_samples = df_samples.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)

        return ds

    @doc(
        summary="""
            Plot a heatmap of ancestry-informative marker (AIM) genotypes.
        """,
        parameters=dict(
            sort="""
                If true (default), sort the samples by the total fraction of AIM
                alleles for the second species in the comparison.
            """,
            row_height="Height per sample in px.",
            xgap="Creates lines between columns (variants).",
            ygap="Creates lines between rows (samples).",
        ),
    )
    def plot_aim_heatmap(
        self,
        aims: aim_params.aims,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sort: bool = True,
        row_height: int = 4,
        xgap: float = 0,
        ygap: float = 0.5,
        palette: Optional[aim_params.palette] = None,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
    ) -> plotly_params.figure:
        # Load AIM calls.
        ds = self.aim_calls(
            aims=aims,
            sample_sets=sample_sets,
            sample_query=sample_query,
        ).compute()
        samples = ds["sample_id"].values
        variant_contig = ds["variant_contig"].values
        contigs = ds.attrs["contigs"]

        # Count variants per contig, needed to figure out how wide to make
        # the columns in the grid of subplots.
        col_widths = [
            np.count_nonzero(variant_contig == contigs.index(contig))
            for contig in contigs
        ]

        # Access and transform genotypes.
        gt = allel.GenotypeArray(ds["call_genotype"].values)
        gn = gt.to_n_alt(fill=-1)

        if sort:
            # Sort by AIM fraction.
            ac = np.sum(gt == 1, axis=(0, 2))
            an = np.sum(gt >= 0, axis=(0, 2))
            af = ac / an
            ix_sorted = np.argsort(af)
            gn = np.take(gn, ix_sorted, axis=1)
            samples = np.take(samples, ix_sorted, axis=0)

        # Set up colors for genotypes
        if palette is None:
            assert self._aim_palettes is not None
            palette = self._aim_palettes[aims]
            assert len(palette) == 4
            # Expect 4 colors, in the order:
            # missing, hom taxon 1, het, hom taxon 2
        species = aims.split("_vs_")

        # Create subplots.
        fig = go_make_subplots(
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

        # Define a discrete color scale.
        # https://plotly.com/python/colorscales/#constructing-a-discrete-or-discontinuous-color-scale
        colorscale = [
            [0 / 4, palette[0]],
            [1 / 4, palette[0]],
            [1 / 4, palette[1]],
            [2 / 4, palette[1]],
            [2 / 4, palette[2]],
            [3 / 4, palette[2]],
            [3 / 4, palette[3]],
            [4 / 4, palette[3]],
        ]

        # Define a colorbar.
        colorbar = dict(
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
        )

        # Define hover text template.
        hovertemplate = dedent(
            """
            Variant index: %{x}<br>
            Sample: %{y}<br>
            Genotype: %{z}
            <extra></extra>
        """
        )

        # Create the subplots, one for each contig.
        for j in range(len(contigs)):
            loc_contig = variant_contig == j
            gn_contig = np.compress(loc_contig, gn, axis=0)
            fig.add_trace(
                go.Heatmap(
                    y=samples,
                    z=gn_contig.T,
                    colorscale=colorscale,
                    zmin=-1.5,
                    zmax=2.5,
                    xgap=xgap,
                    ygap=ygap,  # this creates faint lines between rows
                    colorbar=colorbar,
                    hovertemplate=hovertemplate,
                ),
                row=1,
                col=j + 1,
            )

        # Tidy up the plot.
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

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig
