from typing import Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numpydoc_decorator import doc

from ..util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    Region,
    check_types,
    da_from_zarr,
    init_zarr_store,
    locate_region,
    parse_multi_region,
    simple_xarray_concat,
)
from . import base_params, hap_params
from .base_params import DEFAULT
from .genome_features import AnophelesGenomeFeaturesData
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata


class AnophelesHapData(
    AnophelesSampleMetadata, AnophelesGenomeFeaturesData, AnophelesGenomeSequenceData
):
    def __init__(
        self,
        default_phasing_analysis: Optional[str] = None,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # These will vary between data resources.
        self._default_phasing_analysis = default_phasing_analysis

        # Set up caches.
        self._cache_haplotypes: Dict = dict()
        self._cache_haplotype_sites: Dict = dict()

    @property
    def phasing_analysis_ids(self) -> Tuple[str, ...]:
        """Identifiers for the different phasing analyses that are available.
        These are values than can be used for the `analysis` parameter in any
        method making using of haplotype data.

        """
        return tuple(self.config.get("PHASING_ANALYSIS_IDS", ()))  # ensure tuple

    def _prep_phasing_analysis_param(self, *, analysis: hap_params.analysis):
        if analysis == DEFAULT:
            # Use whatever is the default phasing analysis for this data resource.
            assert self._default_phasing_analysis is not None
            return self._default_phasing_analysis
        elif analysis in self.phasing_analysis_ids:
            return analysis
        else:
            raise ValueError(
                f"Invalid phasing analysis, must be one of f{self.phasing_analysis_ids}."
            )

    @check_types
    @doc(
        summary="Open haplotype sites zarr.",
        returns="Zarr hierarchy.",
    )
    def open_haplotype_sites(
        self, analysis: hap_params.analysis = DEFAULT
    ) -> zarr.hierarchy.Group:
        analysis = self._prep_phasing_analysis_param(analysis=analysis)
        try:
            return self._cache_haplotype_sites[analysis]
        except KeyError:
            path = f"{self._base_path}/{self._major_version_path}/snp_haplotypes/sites/{analysis}/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_haplotype_sites[analysis] = root
        return root

    def _haplotype_sites_for_contig(
        self, *, contig, analysis, field, inline_array, chunks
    ):
        sites = self.open_haplotype_sites(analysis=analysis)
        arr = sites[f"{contig}/variants/{field}"]
        arr = da_from_zarr(arr, inline_array=inline_array, chunks=chunks)
        return arr

    @check_types
    @doc(
        summary="Open haplotypes zarr.",
        returns="Zarr hierarchy.",
    )
    def open_haplotypes(
        self,
        sample_set: base_params.sample_set,
        analysis: hap_params.analysis = DEFAULT,
    ) -> Optional[zarr.hierarchy.Group]:
        analysis = self._prep_phasing_analysis_param(analysis=analysis)
        try:
            return self._cache_haplotypes[(sample_set, analysis)]
        except KeyError:
            release = self.lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_haplotypes/{sample_set}/{analysis}/zarr"
            store = init_zarr_store(fs=self._fs, path=path)
            # Some sample sets have no data for a given analysis, handle this.
            try:
                root = zarr.open_consolidated(store=store)
            except FileNotFoundError:
                root = None
            self._cache_haplotypes[(sample_set, analysis)] = root
        return root

    def _haplotypes_for_contig(
        self, *, contig, sample_set, analysis, inline_array, chunks
    ):
        debug = self._log.debug

        debug("open zarr")
        root = self.open_haplotypes(sample_set=sample_set, analysis=analysis)
        sites = self.open_haplotype_sites(analysis=analysis)

        debug("variant_position")
        pos = sites[f"{contig}/variants/POS"]

        # some sample sets have no data for a given analysis, handle this
        # TODO consider returning a dataset with 0 length samples dimension instead, would
        # probably simplify a lot of other logic
        if root is None:
            return None

        coords = dict()
        data_vars = dict()

        coords["variant_position"] = (
            [DIM_VARIANT],
            da_from_zarr(pos, inline_array=inline_array, chunks=chunks),
        )

        debug("variant_contig")
        contig_index = self.contigs.index(contig)
        coords["variant_contig"] = (
            [DIM_VARIANT],
            da.full_like(pos, fill_value=contig_index, dtype="u1"),
        )

        debug("variant_allele")
        ref = da_from_zarr(
            sites[f"{contig}/variants/REF"], inline_array=inline_array, chunks=chunks
        )
        alt = da_from_zarr(
            sites[f"{contig}/variants/ALT"], inline_array=inline_array, chunks=chunks
        )
        variant_allele = da.hstack([ref[:, None], alt[:, None]])
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        debug("call_genotype")
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
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
        summary="Access haplotype data.",
        returns="A dataset of haplotypes and associated data.",
    )
    def haplotypes(
        self,
        region: base_params.region,
        analysis: hap_params.analysis = DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        random_seed: base_params.random_seed = 42,
    ) -> Optional[xr.Dataset]:
        debug = self._log.debug

        debug("normalise parameters")
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
        regions: List[Region] = parse_multi_region(self, region)
        del region
        analysis = self._prep_phasing_analysis_param(analysis=analysis)

        debug("build dataset")
        lx = []
        for r in regions:
            ly = []

            for s in sample_sets:
                y = self._haplotypes_for_contig(
                    contig=r.contig,
                    sample_set=s,
                    analysis=analysis,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                if y is not None:
                    ly.append(y)

            if len(ly) == 0:
                debug("early out, no data for given sample sets and analysis")
                return None

            debug("concatenate data from multiple sample sets")
            x = simple_xarray_concat(ly, dim=DIM_SAMPLE)

            debug("handle region")
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        debug("concatenate data from multiple regions")
        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

        debug("handle sample query")
        if sample_query is not None:
            debug("load sample metadata")
            df_samples = self.sample_metadata(sample_sets=sample_sets)

            debug("align sample metadata with haplotypes")
            phased_samples = ds["sample_id"].values.tolist()
            df_samples_phased = (
                df_samples.set_index("sample_id").loc[phased_samples].reset_index()
            )

            debug("apply the query")
            loc_samples = df_samples_phased.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)

        debug("handle cohort size")
        if cohort_size is not None:
            debug("handle cohort size")
            # overrides min and max
            min_cohort_size = cohort_size
            max_cohort_size = cohort_size

        if min_cohort_size is not None:
            debug("handle min cohort size")
            n_samples = ds.dims["samples"]
            if n_samples < min_cohort_size:
                raise ValueError(
                    f"not enough samples ({n_samples}) for minimum cohort size ({min_cohort_size})"
                )

        if max_cohort_size is not None:
            debug("handle max cohort size")
            n_samples = ds.dims["samples"]
            if n_samples > max_cohort_size:
                rng = np.random.default_rng(seed=random_seed)
                loc_downsample = rng.choice(
                    n_samples, size=max_cohort_size, replace=False
                )
                loc_downsample.sort()
                ds = ds.isel(samples=loc_downsample)

        return ds
