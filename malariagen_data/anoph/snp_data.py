from typing import Dict, Optional, Tuple

import zarr
from numpydoc_decorator import doc

from ..util import init_zarr_store
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
