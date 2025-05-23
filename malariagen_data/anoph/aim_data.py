from textwrap import dedent
from typing import Dict, Optional

import allel  # type: ignore
import numpy as np
import plotly.graph_objects as go  # type: ignore
import xarray as xr
from numpydoc_decorator import doc  # type: ignore
from plotly.subplots import make_subplots as go_make_subplots  # type: ignore

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
        returns="""
        A dataset with 2 dimensions: `variants` the number of AIMs sites, and `alleles` which will always be 2, each representing one of the species. It contains 2 coordinates:
        `variant_contig` has `variants` values and contains the chromosome arm of each AIM, and `variant_position` has `variants` values and contains the position of each AIM. It contains 1 data variable:
        `variant_allele` has (`variants`, `allele`) values and contains the discriminating alleles for each AIM.
        """,
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
        A dataset with 4 dimensions:
        `variants` the number of AIMs sites,
        `samples` the number of samples,
        `ploidy` the ploidy (2),
        and `alleles` which will always be 2, each representing one of the species. It contains 3 coordinates:
        `sample_id` has `samples` values and contains the identifier of each sample,
        `variant_contig` has `variants` values and contains the chromosome arm of each AIM,
        and `variant_position` has `variants` values and contains the position of each AIM. It contains 2 data variables:
        `call_genotype` has (`variants`, `samples`, `ploidy`) values and contains both calls for each sample and each AIM,
        `variant_allele` has (`variants`, `allele`) values and contains the discriminating alleles for each AIM.
        """,
    )
    def aim_calls(
        self,
        aims: aim_params.aims,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
    ) -> xr.Dataset:
        self._require_aim_analysis()

        # Prepare parameters.
        prepared_aims = self._prep_aims_param(aims=aims)
        del aims
        prepared_sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
        del sample_sets
        prepared_sample_query = self._prep_sample_query_param(sample_query=sample_query)
        del sample_query

        # Start a list of AIM calls Datasets, one for each sample set.
        aim_calls_datasets = []

        # For each sample set...
        for sample_set in prepared_sample_sets:
            # Get the AIM calls for all samples in the set, as a Xarray Dataset.
            aim_calls_dataset = self._aim_calls_dataset(
                aims=prepared_aims,
                sample_set=sample_set,
            )

            # Add this Dataset to the list.
            aim_calls_datasets.append(aim_calls_dataset)

        # Concatenate data from multiple sample sets.
        ds = simple_xarray_concat(aim_calls_datasets, dim=DIM_SAMPLE)

        # If there's a sample query...
        if prepared_sample_query is not None:
            # Get the relevant sample metadata.
            df_samples = self.sample_metadata(sample_sets=prepared_sample_sets)

            # If there are no sample query options, then default to an empty dict.
            sample_query_options = sample_query_options or {}

            # Determine which samples match the sample query.
            loc_samples = df_samples.eval(
                prepared_sample_query, **sample_query_options
            ).values

            # Raise an error if no samples match the sample query.
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(
                    f"No samples found for query {prepared_sample_query!r}"
                )

            # Get the relevant sample ids from the sample metadata DataFrame, using the boolean mask.
            relevant_sample_ids = df_samples.loc[loc_samples, "sample_id"].values

            # Get all the sample ids from the unfiltered AIM calls Dataset.
            ds_sample_ids = ds.coords["sample_id"].values

            # Get the indices of samples in the AIM calls Dataset that match the relevant sample ids.
            # Note: we use `[0]` to get the first element of the tuple returned by `np.where`.
            relevant_sample_indices = np.where(
                np.isin(ds_sample_ids, relevant_sample_ids)
            )[0]

            # Select only the relevant samples from the AIM calls Dataset.
            ds = ds.isel(samples=relevant_sample_indices)

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
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
            sample_query_options=sample_query_options,
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
