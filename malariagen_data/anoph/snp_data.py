from typing import Dict, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from numpydoc_decorator import doc

from ..util import (  # DIM_PLOIDY,; DIM_SAMPLE,
    DIM_ALLELE,
    DIM_VARIANT,
    Region,
    da_compress,
    da_concat,
    da_from_zarr,
    dask_compress_dataset,
    init_zarr_store,
    locate_region,
    parse_region,
    resolve_regions,
    xarray_concat,
)
from .base import DEFAULT, base_params
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata


class AnophelesSnpData(AnophelesSampleMetadata, AnophelesGenomeSequenceData):
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
        contig: base_params.contig,
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

    def _snp_genotypes_for_contig(
        self,
        *,
        contig: base_params.contig,
        sample_set: base_params.sample_set,
        field: base_params.field,
        inline_array: base_params.inline_array,
        chunks: base_params.chunks,
    ) -> da.Array:
        """Access SNP genotypes for a single contig and a single sample set."""
        root = self.open_snp_genotypes(sample_set=sample_set)
        z = root[f"{contig}/calldata/{field}"]
        d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        return d

    @doc(
        summary="Access SNP genotypes and associated data.",
        returns="""
            An array of either genotypes (GT), genotype quality (GQ), allele
            depths (AD) or mapping quality (MQ) values.
        """,
    )
    def snp_genotypes(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        field: base_params.field = "GT",
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> da.Array:
        # Normalise parameters.
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
        regions = resolve_regions(self, region)
        del region

        # Concatenate multiple sample sets and/or contigs.
        lx = []
        for r in regions:
            contig = r.contig
            ly = []

            for s in sample_sets:
                y = self._snp_genotypes_for_contig(
                    contig=contig,
                    sample_set=s,
                    field=field,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            # Concatenate data from multiple sample sets.
            x = da_concat(ly, axis=1)

            # Locate region - do this only once, optimisation.
            if r.start or r.end:
                pos = self._snp_sites_for_contig(
                    contig=contig, field="POS", inline_array=inline_array, chunks=chunks
                )
                loc_region = locate_region(r, pos)
                x = x[loc_region]

            lx.append(x)

        # Concatenate data from multiple regions.
        d = da_concat(lx, axis=0)

        # Apply site filters if requested.
        if site_mask is not None:
            loc_sites = self.site_filters(
                region=regions,
                mask=site_mask,
            )
            d = da_compress(loc_sites, d, axis=0)

        # Apply sample query if requested.
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            d = da.compress(loc_samples, d, axis=1)

        return d

    def _snp_variants_for_contig(
        self,
        *,
        contig: base_params.contig,
        inline_array: base_params.inline_array,
        chunks: base_params.chunks,
    ):
        coords = dict()
        data_vars = dict()
        sites_root = self.open_snp_sites()

        # Set up variant_position.
        pos_z = sites_root[f"{contig}/variants/POS"]
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        # Set up variant_allele.
        ref_z = sites_root[f"{contig}/variants/REF"]
        alt_z = sites_root[f"{contig}/variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # Set up variant_contig.
        contig_index = self.contigs.index(contig)
        variant_contig = da.full_like(
            variant_position, fill_value=contig_index, dtype="u1"
        )
        coords["variant_contig"] = [DIM_VARIANT], variant_contig

        # Set up site filters arrays.
        for mask in self.site_mask_ids:
            filters_root = self.open_site_filters(mask=mask)
            z = filters_root[f"{contig}/variants/filter_pass"]
            d = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            data_vars[f"variant_filter_pass_{mask}"] = [DIM_VARIANT], d

        # Set up attributes.
        attrs = {"contigs": self.contigs}

        # Create a dataset.
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        return ds

    @doc(
        summary="Access SNP sites and site filters.",
        returns="A dataset containing SNP sites and site filters.",
    )
    def snp_variants(
        self,
        region: base_params.region,
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ):
        # Normalise parameters.
        regions = resolve_regions(self, region)
        del region

        # Access SNP data and concatenate multiple regions.
        lx = []
        for r in regions:
            # Access variants.
            x = self._snp_variants_for_contig(
                contig=r.contig,
                inline_array=inline_array,
                chunks=chunks,
            )

            # Handle region.
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        # Concatenate data from multiple regions.
        ds = xarray_concat(lx, dim=DIM_VARIANT)

        # Apply site filters.
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        return ds

    @doc(
        summary="Compute genome accessibility array.",
        returns="An array of boolean values identifying accessible genome sites.",
    )
    def is_accessible(
        self,
        region: base_params.region,
        site_mask: base_params.site_mask = DEFAULT,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> np.ndarray:
        resolved_region = parse_region(self, region)
        del region

        # Determine contig sequence length.
        seq_length = self.genome_sequence(resolved_region).shape[0]

        # Set up output.
        is_accessible = np.zeros(seq_length, dtype=bool)

        # Access SNP site positions.
        pos = self.snp_sites(region=resolved_region, field="POS").compute()
        if resolved_region.start:
            offset = resolved_region.start
        else:
            offset = 1

        # Access site filters.
        filter_pass = self._site_filters_for_region(
            region=resolved_region,
            mask=site_mask,
            field="filter_pass",
            inline_array=inline_array,
            chunks=chunks,
        ).compute()

        # Assign values from site filters.
        is_accessible[pos - offset] = filter_pass

        return is_accessible
