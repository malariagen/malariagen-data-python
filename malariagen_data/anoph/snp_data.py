from typing import Dict, List, Optional, Tuple, Union

import allel
import bokeh
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numpydoc_decorator import doc

from ..util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    CacheMiss,
    Region,
    check_types,
    da_compress,
    da_concat,
    da_from_zarr,
    dask_compress_dataset,
    init_zarr_store,
    locate_region,
    parse_multi_region,
    parse_single_region,
    simple_xarray_concat,
    true_runs,
)
from . import base_params
from .base_params import DEFAULT
from .genome_features import AnophelesGenomeFeaturesData, gplt_params
from .genome_sequence import AnophelesGenomeSequenceData
from .sample_metadata import AnophelesSampleMetadata


class AnophelesSnpData(
    AnophelesSampleMetadata, AnophelesGenomeFeaturesData, AnophelesGenomeSequenceData
):
    def __init__(
        self,
        site_filters_analysis: Optional[str] = None,
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
        self._default_site_mask = default_site_mask

        # Set up caches.
        # TODO review type annotations here, maybe can tighten
        self._cache_snp_sites = None
        self._cache_snp_genotypes: Dict = dict()
        self._cache_site_filters: Dict = dict()
        self._cache_site_annotations = None
        self._cache_locate_site_class: Dict = dict()

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
        return tuple(self.config.get("SITE_MASK_IDS", ()))  # ensure tuple

    @property
    def _site_annotations_zarr_path(self) -> str:
        return self.config["SITE_ANNOTATIONS_ZARR_PATH"]

    def _prep_site_mask_param(
        self,
        *,
        site_mask: base_params.site_mask,
    ) -> base_params.site_mask:
        if site_mask == DEFAULT:
            # Use whatever is the default site mask for this data resource.
            assert self._default_site_mask is not None
            return self._default_site_mask
        elif site_mask in self.site_mask_ids:
            return site_mask
        else:
            raise ValueError(
                f"Invalid site mask, must be one of f{self.site_mask_ids}."
            )

    def _prep_optional_site_mask_param(
        self,
        *,
        site_mask: Optional[base_params.site_mask],
    ) -> Optional[base_params.site_mask]:
        if site_mask is None:
            # This is allowed, it means don't apply any site mask to the data.
            return None
        else:
            return self._prep_site_mask_param(site_mask=site_mask)

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

    @check_types
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

    @check_types
    @doc(
        summary="Open site filters zarr.",
        returns="Zarr hierarchy.",
    )
    def open_site_filters(
        self,
        mask: base_params.site_mask,
    ) -> zarr.hierarchy.Group:
        self._require_site_filters_analysis()
        mask = self._prep_site_mask_param(site_mask=mask)

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

    @doc(
        summary="Open site annotations zarr.",
        returns="Zarr hierarchy.",
    )
    def open_site_annotations(self) -> zarr.hierarchy.Group:
        if self._cache_site_annotations is None:
            path = f"{self._base_path}/{self._site_annotations_zarr_path}"
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_site_annotations = zarr.open_consolidated(store=store)
        return self._cache_site_annotations

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

    @check_types
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
        regions: List[Region] = parse_multi_region(self, region)
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
    ) -> da.Array:
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
            loc_region = locate_region(region, np.asarray(pos))
            ret = ret[loc_region]

        return ret

    @check_types
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
        regions: List[Region] = parse_multi_region(self, region)
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

    @check_types
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
        sample_indices: Optional[base_params.sample_indices] = None,
        field: base_params.field = "GT",
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> da.Array:
        # Additional parameter checks.
        base_params.validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Normalise parameters.
        sample_sets = self._prep_sample_sets_param(sample_sets=sample_sets)
        regions: List[Region] = parse_multi_region(self, region)
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
                loc_region = locate_region(r, np.asarray(pos))
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

        # Apply sample selection if requested.
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            loc_samples = df_samples.eval(sample_query).values
            d = da.compress(loc_samples, d, axis=1)
        elif sample_indices is not None:
            d = da.take(d, sample_indices, axis=1)

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

    @check_types
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
        regions: List[Region] = parse_multi_region(self, region)
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
        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

        # Apply site filters.
        if site_mask is not None:
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        return ds

    @check_types
    @doc(
        summary="Load site annotations.",
        returns="A dataset of site annotations.",
    )
    def site_annotations(
        self,
        region: base_params.single_region,
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> xr.Dataset:
        # N.B., we default to chunks="auto" here for performance reasons

        # Resolve region.
        resolved_region: Region = parse_single_region(self, region)
        del region
        contig = resolved_region.contig

        # Open site annotations zarr.
        root = self.open_site_annotations()

        # Build a dataset.
        ds = xr.Dataset()
        for field in (
            "codon_degeneracy",
            "codon_nonsyn",
            "codon_position",
            "seq_cls",
            "seq_flen",
            "seq_relpos_start",
            "seq_relpos_stop",
        ):
            data = da_from_zarr(
                root[field][contig],
                inline_array=inline_array,
                chunks=chunks,
            )
            ds[field] = "variants", data

        # Subset to SNP positions.
        pos = self.snp_sites(
            region=contig,
            field="POS",
            site_mask=site_mask,
            inline_array=inline_array,
            chunks=chunks,
        )
        pos = pos.compute()
        if resolved_region.start or resolved_region.end:
            loc_region = locate_region(resolved_region, pos)
            pos = pos[loc_region]
        idx = pos - 1
        ds = ds.isel(variants=idx)

        return ds

    def _locate_site_class(
        self,
        *,
        region: base_params.single_region,
        site_mask: Optional[base_params.site_mask],
        site_class: base_params.site_class,
    ):
        # Cache these data in memory to avoid repeated computation.
        cache_key = (region, site_mask, site_class)

        try:
            loc_ann = self._cache_locate_site_class[cache_key]

        except KeyError:
            # Access site annotations data.
            ds_ann = self.site_annotations(
                region=region,
                site_mask=site_mask,
            )
            codon_pos = ds_ann["codon_position"].data
            codon_deg = ds_ann["codon_degeneracy"].data
            seq_cls = ds_ann["seq_cls"].data
            seq_flen = ds_ann["seq_flen"].data
            seq_relpos_start = ds_ann["seq_relpos_start"].data
            seq_relpos_stop = ds_ann["seq_relpos_stop"].data
            site_class = site_class.upper()

            # Define constants used in site annotations data.
            SEQ_CLS_UNKNOWN = 0  # noqa
            SEQ_CLS_UPSTREAM = 1
            SEQ_CLS_DOWNSTREAM = 2
            SEQ_CLS_5UTR = 3
            SEQ_CLS_3UTR = 4
            SEQ_CLS_CDS_FIRST = 5
            SEQ_CLS_CDS_MID = 6
            SEQ_CLS_CDS_LAST = 7
            SEQ_CLS_INTRON_FIRST = 8
            SEQ_CLS_INTRON_MID = 9
            SEQ_CLS_INTRON_LAST = 10
            CODON_DEG_UNKNOWN = 0  # noqa
            CODON_DEG_0 = 1
            CODON_DEG_2_SIMPLE = 2
            CODON_DEG_2_COMPLEX = 3  # noqa
            CODON_DEG_4 = 4

            # Set up site selection.

            if site_class == "CDS_DEG_4":
                # 4-fold degenerate coding sites
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_CDS_FIRST)
                        | (seq_cls == SEQ_CLS_CDS_MID)
                        | (seq_cls == SEQ_CLS_CDS_LAST)
                    )
                    & (codon_pos == 2)
                    & (codon_deg == CODON_DEG_4)
                )

            elif site_class == "CDS_DEG_2_SIMPLE":
                # 2-fold degenerate coding sites
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_CDS_FIRST)
                        | (seq_cls == SEQ_CLS_CDS_MID)
                        | (seq_cls == SEQ_CLS_CDS_LAST)
                    )
                    & (codon_pos == 2)
                    & (codon_deg == CODON_DEG_2_SIMPLE)
                )

            elif site_class == "CDS_DEG_0":
                # non-degenerate coding sites
                loc_ann = (
                    (seq_cls == SEQ_CLS_CDS_FIRST)
                    | (seq_cls == SEQ_CLS_CDS_MID)
                    | (seq_cls == SEQ_CLS_CDS_LAST)
                ) & (codon_deg == CODON_DEG_0)

            elif site_class == "INTRON_SHORT":
                # short introns, excluding splice regions
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_INTRON_FIRST)
                        | (seq_cls == SEQ_CLS_INTRON_MID)
                        | (seq_cls == SEQ_CLS_INTRON_LAST)
                    )
                    & (seq_flen < 100)
                    & (seq_relpos_start > 10)
                    & (seq_relpos_stop > 10)
                )

            elif site_class == "INTRON_LONG":
                # long introns, excluding splice regions
                loc_ann = (
                    (
                        (seq_cls == SEQ_CLS_INTRON_FIRST)
                        | (seq_cls == SEQ_CLS_INTRON_MID)
                        | (seq_cls == SEQ_CLS_INTRON_LAST)
                    )
                    & (seq_flen > 200)
                    & (seq_relpos_start > 10)
                    & (seq_relpos_stop > 10)
                )

            elif site_class == "INTRON_SPLICE_5PRIME":
                # 5' intron splice regions
                loc_ann = (
                    (seq_cls == SEQ_CLS_INTRON_FIRST)
                    | (seq_cls == SEQ_CLS_INTRON_MID)
                    | (seq_cls == SEQ_CLS_INTRON_LAST)
                ) & (seq_relpos_start < 2)

            elif site_class == "INTRON_SPLICE_3PRIME":
                # 3' intron splice regions
                loc_ann = (
                    (seq_cls == SEQ_CLS_INTRON_FIRST)
                    | (seq_cls == SEQ_CLS_INTRON_MID)
                    | (seq_cls == SEQ_CLS_INTRON_LAST)
                ) & (seq_relpos_stop < 2)

            elif site_class == "UTR_5PRIME":
                # 5' UTR
                loc_ann = seq_cls == SEQ_CLS_5UTR

            elif site_class == "UTR_3PRIME":
                # 3' UTR
                loc_ann = seq_cls == SEQ_CLS_3UTR

            elif site_class == "INTERGENIC":
                # intergenic regions, distant from a gene
                loc_ann = (
                    (seq_cls == SEQ_CLS_UPSTREAM) & (seq_relpos_stop > 10_000)
                ) | ((seq_cls == SEQ_CLS_DOWNSTREAM) & (seq_relpos_start > 10_000))

            else:
                raise NotImplementedError(site_class)

            # Compute site selection.
            with self._dask_progress(desc=f"Locate {site_class} sites"):
                loc_ann = loc_ann.compute()

            self._cache_locate_site_class[cache_key] = loc_ann

        return loc_ann

    def _snp_calls_for_contig(
        self,
        *,
        contig: base_params.contig,
        sample_set: base_params.sample_set,
        inline_array: base_params.inline_array,
        chunks: base_params.chunks,
    ) -> xr.Dataset:
        coords = dict()
        data_vars = dict()

        # Set up call arrays.
        calls_root = self.open_snp_genotypes(sample_set=sample_set)
        gt_z = calls_root[f"{contig}/calldata/GT"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        gq_z = calls_root[f"{contig}/calldata/GQ"]
        call_gq = da_from_zarr(gq_z, inline_array=inline_array, chunks=chunks)
        ad_z = calls_root[f"{contig}/calldata/AD"]
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        mq_z = calls_root[f"{contig}/calldata/MQ"]
        call_mq = da_from_zarr(mq_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_GQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_gq)
        data_vars["call_MQ"] = ([DIM_VARIANT, DIM_SAMPLE], call_mq)
        data_vars["call_AD"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE],
            call_ad,
        )

        # Set up sample arrays.
        z = calls_root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        # Decode to unicode strings, as it is stored as bytes objects.
        sample_id = sample_id.astype("U")
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        # Create a dataset.
        ds = xr.Dataset(data_vars=data_vars, coords=coords)

        return ds

    @check_types
    @doc(
        summary="Access SNP sites, site filters and genotype calls.",
        returns="A dataset containing SNP sites, site filters and genotype calls.",
    )
    def snp_calls(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = None,
        site_class: Optional[base_params.site_class] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        random_seed: base_params.random_seed = 42,
    ) -> xr.Dataset:
        # Additional parameter checks.
        base_params.validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Normalise parameters.
        sample_sets_prepped: List[str] = self._prep_sample_sets_param(
            sample_sets=sample_sets
        )
        del sample_sets
        regions: List[Region] = parse_multi_region(self, region)
        del region

        # Access SNP calls and concatenate multiple sample sets and/or regions.
        lx = []
        for r in regions:
            ly = []
            for s in sample_sets_prepped:
                y = self._snp_calls_for_contig(
                    contig=r.contig,
                    sample_set=s,
                    inline_array=inline_array,
                    chunks=chunks,
                )
                ly.append(y)

            # Concatenate data from multiple sample sets.
            x = simple_xarray_concat(ly, dim=DIM_SAMPLE)

            # Add variants variables.
            v = self._snp_variants_for_contig(
                contig=r.contig, inline_array=inline_array, chunks=chunks
            )
            x = xr.merge([v, x], compat="override", join="override")

            # Handle site class.
            if site_class is not None:
                loc_ann = self._locate_site_class(
                    region=r.contig,
                    site_class=site_class,
                    site_mask=None,
                )
                x = x.isel(variants=loc_ann)

            # Handle region, do this only once - optimisation.
            if r.start or r.end:
                pos = x["variant_position"].values
                loc_region = locate_region(r, pos)
                x = x.isel(variants=loc_region)

            lx.append(x)

        # Concatenate data from multiple regions.
        ds = simple_xarray_concat(lx, dim=DIM_VARIANT)

        if site_mask is not None:
            # Apply site filters.
            ds = dask_compress_dataset(
                ds, indexer=f"variant_filter_pass_{site_mask}", dim=DIM_VARIANT
            )

        # Add call_genotype_mask.
        ds["call_genotype_mask"] = ds["call_genotype"] < 0

        # Handle sample selection.
        if sample_query is not None:
            df_samples = self.sample_metadata(sample_sets=sample_sets_prepped)
            loc_samples = df_samples.eval(sample_query).values
            if np.count_nonzero(loc_samples) == 0:
                raise ValueError(f"No samples found for query {sample_query!r}")
            ds = ds.isel(samples=loc_samples)
        elif sample_indices is not None:
            ds = ds.isel(samples=sample_indices)

        # Handle cohort size, overrides min and max.
        if cohort_size is not None:
            min_cohort_size = cohort_size
            max_cohort_size = cohort_size

        # Handle min cohort size.
        if min_cohort_size is not None:
            n_samples = ds.dims["samples"]
            if n_samples < min_cohort_size:
                raise ValueError(
                    f"not enough samples ({n_samples}) for minimum cohort size ({min_cohort_size})"
                )

        # Handle max cohort size.
        if max_cohort_size is not None:
            n_samples = ds.dims["samples"]
            if n_samples > max_cohort_size:
                rng = np.random.default_rng(seed=random_seed)
                loc_downsample = rng.choice(
                    n_samples, size=max_cohort_size, replace=False
                )
                loc_downsample.sort()
                ds = ds.isel(samples=loc_downsample)

        return ds

    def snp_dataset(self, *args, **kwargs):
        """Deprecated, this method has been renamed to snp_calls()."""
        return self.snp_calls(*args, **kwargs)

    def _prep_region_cache_param(
        self, *, region: base_params.region
    ) -> Union[dict, List[dict]]:
        """Obtain a normalised representation of a region parameter which can
        be used with the results cache."""

        # N.B., we need to convert to a dict, because cache saves params as
        # JSON.

        region_prepped: List[Region] = parse_multi_region(self, region)
        if len(region_prepped) > 1:
            ret = [r.to_dict() for r in region_prepped]
        else:
            ret = region_prepped[0].to_dict()
        return ret

    def _results_cache_add_analysis_params(self, params: dict):
        super()._results_cache_add_analysis_params(params)
        params["site_filters_analysis"] = self._site_filters_analysis

    def _snp_allele_counts(
        self,
        *,
        region,
        sample_sets,
        sample_indices,
        site_mask,
        site_class,
        cohort_size,
        random_seed,
    ):
        # Access SNP calls.
        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )
        gt = ds_snps["call_genotype"]

        # Set up and run allele counts computation.
        gt = allel.GenotypeDaskArray(gt.data)
        ac = gt.count_alleles(max_allele=3)
        with self._dask_progress(desc="Compute SNP allele counts"):
            ac = ac.compute()

        # Return plain numpy array.
        results = dict(ac=ac.values)

        return results

    @check_types
    @doc(
        summary="""
            Compute SNP allele counts. This returns the number of times each
            SNP allele was observed in the selected samples.
        """,
        returns="""
            A numpy array of shape (n_variants, 4), where the first column has
            the reference allele (0) counts, the second column has the first
            alternate allele (1) counts, the third column has the second
            alternate allele (2) counts, and the fourth column has the third
            alternate allele (3) counts.
        """,
        notes="""
            This computation may take some time to run, depending on your
            computing environment. Results of this computation will be cached
            and re-used if the `results_cache` parameter was set when
            instantiating the class.
        """,
    )
    def snp_allele_counts(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = None,
        site_class: Optional[base_params.site_class] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
    ) -> np.ndarray:
        # Change this name if you ever change the behaviour of this function,
        # to invalidate any previously cached data.
        name = "snp_allele_counts_v2"

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
            sample_sets=sample_sets_prepped,
            sample_indices=sample_indices_prepped,
            site_mask=site_mask_prepped,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._snp_allele_counts(**params)
            self.results_cache_set(name=name, params=params, results=results)

        ac = results["ac"]
        return ac

    @check_types
    @doc(
        summary="""
            Plot SNPs in a given genome region. SNPs are shown as rectangles,
            with segregating and non-segregating SNPs positioned on different levels,
            and coloured by site filter.
        """,
        parameters=dict(
            max_snps="Maximum number of SNPs to show.",
        ),
    )
    def plot_snps(
        self,
        region: base_params.region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        site_mask: base_params.site_mask = DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.height = 80,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        max_snps: int = 200_000,
        show: gplt_params.show = True,
    ) -> gplt_params.figure:
        # Plot SNPs track.
        fig1 = self.plot_snps_track(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
            cohort_size=cohort_size,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            max_snps=max_snps,
            show=False,
        )
        fig1.xaxis.visible = False

        # Plot genes track.
        fig2 = self.plot_genes(
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
        )

        # Layout tracks in a grid.
        fig = bokeh.layouts.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Plot SNPs in a given genome region. SNPs are shown as rectangles,
            with segregating and non-segregating SNPs positioned on different levels,
            and coloured by site filter.
        """,
        parameters=dict(
            max_snps="Maximum number of SNPs to show.",
        ),
    )
    def plot_snps_track(
        self,
        region: base_params.single_region,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        site_mask: base_params.site_mask = DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 120,
        max_snps: int = 200_000,
        x_range: Optional[gplt_params.x_range] = None,
        show: gplt_params.show = True,
    ) -> gplt_params.figure:
        site_mask = self._prep_site_mask_param(site_mask=site_mask)

        # Resolve and check region.
        resolved_region: Region = parse_single_region(self, region)
        del region

        if (
            (resolved_region.start is None)
            or (resolved_region.end is None)
            or ((resolved_region.end - resolved_region.start) > max_snps)
        ):
            raise ValueError("Region is too large, please provide a smaller region.")

        # Compute allele counts.
        ac = allel.AlleleCountsArray(
            self.snp_allele_counts(
                region=resolved_region,
                sample_sets=sample_sets,
                sample_query=sample_query,
                site_mask=None,
                cohort_size=cohort_size,
            )
        )
        an = ac.sum(axis=1)
        is_seg = ac.is_segregating()
        is_var = ac.is_variant()
        allelism = ac.allelism()

        # Obtain SNP variants data.
        ds_sites = self.snp_variants(
            region=resolved_region,
        ).compute()

        # Build a dataframe.
        pos = ds_sites["variant_position"].values
        alleles = ds_sites["variant_allele"].values.astype("U")
        cols = {
            "pos": pos,
            "allele_0": alleles[:, 0],
            "allele_1": alleles[:, 1],
            "allele_2": alleles[:, 2],
            "allele_3": alleles[:, 3],
            "ac_0": ac[:, 0],
            "ac_1": ac[:, 1],
            "ac_2": ac[:, 2],
            "ac_3": ac[:, 3],
            "an": an,
            "is_seg": is_seg,
            "is_var": is_var,
            "allelism": allelism,
        }

        for site_mask_id in self.site_mask_ids:
            cols[f"pass_{site_mask_id}"] = ds_sites[
                f"variant_filter_pass_{site_mask_id}"
            ].values

        data = pd.DataFrame(cols)

        # Find gaps in the reference genome.
        seq = self.genome_sequence(region=resolved_region.contig).compute()
        is_n = (seq == b"N") | (seq == b"n")
        n_starts, n_stops = true_runs(is_n)

        # Create figure.
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        pos = data["pos"].values
        x_min = resolved_region.start or 1
        x_max = resolved_region.end or len(seq)
        if x_range is None:
            x_range = bokeh.models.Range1d(x_min, x_max, bounds="auto")

        tooltips = [
            ("Position", "$x{0,0}"),
            (
                "Alleles",
                "@allele_0 (@ac_0), @allele_1 (@ac_1), @allele_2 (@ac_2), @allele_3 (@ac_3)",
            ),
            ("No. alleles", "@allelism"),
            ("Allele calls", "@an"),
        ]

        for site_mask_id in self.site_mask_ids:
            tooltips.append((f"Pass {site_mask_id}", f"@pass_{site_mask_id}"))

        fig = bokeh.plotting.figure(
            title="SNPs",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0.5, 2.5),
            tooltips=tooltips,
        )
        hover_tool = fig.select(type=bokeh.models.HoverTool)
        hover_tool.name = "snps"

        # Plot gaps in the reference genome.
        df_n_runs = pd.DataFrame(
            {"left": n_starts + 0.6, "right": n_stops + 0.4, "top": 2.5, "bottom": 0.5}
        )
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            color="#cccccc",
            source=df_n_runs,
            name="gaps",
        )

        # Plot SNPs.
        color_pass = bokeh.palettes.Colorblind6[3]
        color_fail = bokeh.palettes.Colorblind6[5]
        data["left"] = data["pos"] - 0.4
        data["right"] = data["pos"] + 0.4
        data["bottom"] = np.where(data["is_seg"], 1.6, 0.6)
        data["top"] = data["bottom"] + 0.8
        data["color"] = np.where(data[f"pass_{site_mask}"], color_pass, color_fail)
        fig.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            color="color",
            source=data,
            name="snps",
        )

        # Tidy plot.
        fig.yaxis.ticker = bokeh.models.FixedTicker(
            ticks=[1, 2],
        )
        fig.yaxis.major_label_overrides = {
            1: "Non-segregating",
            2: "Segregating",
        }
        fig.xaxis.axis_label = f"Contig {resolved_region.contig} position (bp)"
        fig.xaxis.ticker = bokeh.models.AdaptiveTicker(min_interval=1)
        fig.xaxis.minor_tick_line_color = None
        fig.xaxis[0].formatter = bokeh.models.NumeralTickFormatter(format="0,0")

        if show:
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Compute genome accessibility array.",
        returns="An array of boolean values identifying accessible genome sites.",
    )
    def is_accessible(
        self,
        region: base_params.single_region,
        site_mask: base_params.site_mask = DEFAULT,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> np.ndarray:
        resolved_region: Region = parse_single_region(self, region)
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
