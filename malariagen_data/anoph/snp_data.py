from typing import Dict, Optional, Tuple

import dask.array as da
import zarr
from numpydoc_decorator import doc

from ..util import (
    Region,
    da_compress,
    da_concat,
    da_from_zarr,
    init_zarr_store,
    locate_region,
    resolve_regions,
)
from .base import DEFAULT, base_params
from .sample_metadata import AnophelesSampleMetadata


class AnophelesSnpData(AnophelesSampleMetadata):
    def __init__(
        self,
        site_filters_analysis: Optional[str] = None,
        site_mask_ids: Optional[Tuple[str, ...]] = None,
        default_site_mask: Optional[str] = None,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # If provided, this analysis version will override the
        # default value provided in the release configuration.
        self._site_filters_analysis_override = site_filters_analysis

        # These will vary between data resources.
        self._site_mask_ids: Tuple[str, ...] = site_mask_ids or ()  # ensure tuple
        self._default_site_mask = default_site_mask

        # Set up caches.
        self._cache_snp_sites = None
        self._cache_snp_genotypes: Dict = dict()
        self._cache_site_filters: Dict = dict()

    @property
    def _site_filters_analysis(self) -> Optional[str]:
        if self._site_filters_analysis_override:
            return self._site_filters_analysis_override
        else:
            # N.B., this will return None if the key is not present in the
            # config.
            return self.config.get("DEFAULT_SITE_FILTERS_ANALYSIS")

    @property
    def site_mask_ids(self) -> Tuple[str, ...]:
        """Identifiers for the different site masks that are available.
        These are values than can be used for the `site_mask` parameter in any
        method making using of SNP data.

        """
        return self._site_mask_ids

    def _prep_site_mask_param(self, *, site_mask: Optional[base_params.site_mask]):
        if site_mask is None:
            # This is allowed, it means don't apply any site mask to the data.
            return None
        elif site_mask == DEFAULT:
            # Use whatever is the default site mask for this data resource.
            return self._default_site_mask
        elif site_mask in self.site_mask_ids:
            return site_mask
        else:
            raise ValueError(
                f"Invalid site mask, must be one of f{self.site_mask_ids}."
            )

    @doc(
        summary="Open SNP sites zarr",
        returns="Zarr hierarchy.",
    )
    def open_snp_sites(self) -> zarr.hierarchy.Group:
        # Here we cache the opened zarr hierarchy, to avoid small delays
        # reading zarr metadata.
        if self._cache_snp_sites is None:
            path = (
                f"{self._base_path}/{self._major_version_path}/snp_genotypes/all/sites/"
            )
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    @doc(
        summary="Open SNP genotypes zarr for a given sample set.",
        returns="Zarr hierarchy.",
    )
    def open_snp_genotypes(
        self, sample_set: base_params.sample_set
    ) -> zarr.hierarchy.Group:
        # Here we cache the opened zarr hierarchy, to avoid small delays
        # reading zarr metadata.
        try:
            return self._cache_snp_genotypes[sample_set]
        except KeyError:
            release = self.lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release)
            path = f"{self._base_path}/{release_path}/snp_genotypes/all/{sample_set}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_snp_genotypes[sample_set] = root
            return root

    def _require_site_filters_analysis(self):
        if not self._site_filters_analysis:
            raise NotImplementedError(
                "Site filters not available for this data resource."
            )

    @doc(
        summary="Open site filters zarr.",
        returns="Zarr hierarchy.",
    )
    def open_site_filters(self, mask: base_params.site_mask) -> zarr.hierarchy.Group:
        self._require_site_filters_analysis()
        # Here we cache the opened zarr hierarchy, to avoid small delays
        # reading zarr metadata.
        try:
            return self._cache_site_filters[mask]
        except KeyError:
            path = f"{self._base_path}/{self._major_version_path}/site_filters/{self._site_filters_analysis}/{mask}/"
            store = init_zarr_store(fs=self._fs, path=path)
            root = zarr.open_consolidated(store=store)
            self._cache_site_filters[mask] = root
            return root

    def _site_filters_for_region(
        self,
        *,
        region: Region,
        mask: base_params.site_mask,
        field: base_params.field,
        inline_array: base_params.inline_array,
        chunks: base_params.chunks,
    ):
        root = self.open_site_filters(mask=mask)
        z = root[f"{region.contig}/variants/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        if region.start or region.end:
            root = self.open_snp_sites()
            pos = root[f"{region.contig}/variants/POS"][:]
            loc_region = locate_region(region, pos)
            d = d[loc_region]
        return d

    @doc(
        summary="Access SNP site filters.",
        returns="""
            An array of boolean values identifying sites that pass the filters.
        """,
    )
    def site_filters(
        self,
        region: base_params.region,
        mask: base_params.site_mask,
        field: base_params.field = "filter_pass",
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> da.Array:
        # Resolve the region parameter to a standard type.
        regions = resolve_regions(self, region)
        del region

        # Load arrays and concatenate if needed.
        d = da_concat(
            [
                self._site_filters_for_region(
                    region=r,
                    mask=mask,
                    field=field,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                for r in regions
            ]
        )

        return d

    def _snp_sites_for_contig(
        self,
        *,
        contig: str,
        field: base_params.field,
        inline_array: base_params.inline_array,
        chunks: base_params.chunks,
    ):
        """Access SNP sites data for a single contig."""
        root = self.open_snp_sites()
        z = root[f"{contig}/variants/{field}"]
        ret = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        return ret

    def _snp_sites_for_region(
        self,
        *,
        region: Region,
        field: base_params.field,
        inline_array: base_params.inline_array,
        chunks: base_params.chunks,
    ) -> da.Array:
        # Access data for the requested contig.
        ret = self._snp_sites_for_contig(
            contig=region.contig, field=field, inline_array=inline_array, chunks=chunks
        )

        # Deal with a region.
        if region.start or region.end:
            if field == "POS":
                pos = ret
            else:
                pos = self._snp_sites_for_contig(
                    contig=region.contig,
                    field="POS",
                    inline_array=inline_array,
                    chunks=chunks,
                )
            loc_region = locate_region(region, pos)
            ret = ret[loc_region]

        return ret

    @doc(
        summary="Access SNP site data (positions or alleles).",
        returns="""
            An array of either SNP positions ("POS"), reference alleles ("REF") or
            alternate alleles ("ALT").
        """,
    )
    def snp_sites(
        self,
        region: base_params.region,
        field: base_params.field,
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> da.Array:
        # Resolve the region parameter to a standard type.
        regions = resolve_regions(self, region)
        del region

        # Access SNP sites and concatenate over regions.
        ret = da_concat(
            [
                self._snp_sites_for_region(
                    region=r,
                    field=field,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for r in regions
            ],
            axis=0,
        )

        # Apply site mask if requested.
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=regions,
                mask=site_mask,
                chunks=chunks,
                inline_array=inline_array,
            )
            ret = da_compress(loc_sites, ret, axis=0)

        return ret
