from typing import Optional

import allel  # type: ignore
import os

import bed_reader
import xarray as xr
from numpydoc_decorator import doc  # type: ignore

from ..util import _check_types, _hash_params
from .snp_data import AnophelesSnpData
from . import base_params
from . import ld_params
from . import plink_params
from . import pca_params


class PlinkConverter(
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

    def _write_plink(
        self,
        ds: xr.Dataset,
        bed_file_path: str,
    ) -> None:
        """Shared helper: convert a biallelic SNP dataset to PLINK .bed."""
        # Compute genotype ref counts.
        with self._dask_progress("Computing genotype ref counts"):
            gt = ds["call_genotype"].data  # dask array
            gn_ref = allel.GenotypeDaskArray(gt).to_n_ref(fill=-127)
            gn_ref = gn_ref.compute()

        val = gn_ref.T
        with self._spinner("Prepare PLINK output data"):
            alleles = ds["variant_allele"].values
            properties = {
                "iid": ds["sample_id"].values,
                "chromosome": ds["variant_contig"].values,
                "bp_position": ds["variant_position"].values,
                "allele_1": alleles[:, 0],
                "allele_2": alleles[:, 1],
            }

        bed_reader.to_bed(
            filepath=bed_file_path,
            val=val,
            properties=properties,
            count_A1=True,
        )

    @doc(
        summary="""
            Write Anopheles biallelic SNP data to the Plink binary file format.
        """,
        extended_summary="""
            This function writes biallelic SNPs to the Plink binary file format. It enables
            subsetting to specific regions (`region`), selecting specific sample sets, or lists of
            samples, randomly downsampling sites, and specifying filters based on missing data and
            minimum minor allele count (see the docs for `biallelic_snp_calls` for more information).
            The `overwrite` parameter, set to true, will enable overwrite of data with the same
            SNP selection parameter values.
        """,
        returns="""
        Base path to files containing binary Plink output files. Append .bed,
        .bim or .fam to obtain paths for the binary genotype table file, variant
        information file and sample information file respectively.
        """,
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Unless the `overwrite` parameter is set to `True`, results will be returned
            from a previous computation, if available.
        """,
    )
    def biallelic_snps_to_plink(
        self,
        output_dir: plink_params.output_dir,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        overwrite: plink_params.overwrite = False,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        min_minor_ac: Optional[
            base_params.min_minor_ac
        ] = pca_params.min_minor_ac_default,
        max_missing_an: Optional[
            base_params.max_missing_an
        ] = pca_params.max_missing_an_default,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        out: Optional[plink_params.out] = None,
    ):
        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Use user-provided prefix or fall back to auto-generated default
        if out is not None:
            plink_file_path = f"{output_dir}/{out}"
        else:
            plink_file_path = f"{output_dir}/{region}.{n_snps}.{min_minor_ac}.{max_missing_an}.{thin_offset}"

        bed_file_path = f"{plink_file_path}.bed"

        # Check to see if file exists and if overwrite is set to false, return existing file
        if os.path.exists(bed_file_path):
            if not overwrite:
                return plink_file_path

        # Get snps
        ds_snps = self.biallelic_snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_snps=n_snps,
            thin_offset=thin_offset,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        self._write_plink(ds_snps, bed_file_path)

        return plink_file_path

    @_check_types
    @doc(
        summary="""
            Write LD-pruned biallelic SNP data to the PLINK binary file format.
        """,
        extended_summary="""
            This function first performs LD pruning on biallelic SNPs (see
            ``biallelic_snps_ld_pruned``), then writes the pruned data to PLINK
            binary format (.bed, .bim, .fam). The resulting files are suitable
            for use with ADMIXTURE and other population-genetics tools that
            accept PLINK input.

            The ``overwrite`` parameter, set to True, will enable overwrite of
            data with the same parameter values.
        """,
        returns="""
            Base path to files containing binary PLINK output files. Append
            .bed, .bim or .fam to obtain paths for the binary genotype table
            file, variant information file and sample information file
            respectively.
        """,
        notes="""
            This computation may take some time to run, depending on your
            computing environment. Unless the ``overwrite`` parameter is set to
            ``True``, results will be returned from a previous computation, if
            available.
        """,
    )
    def biallelic_snps_ld_pruned_to_plink(
        self,
        output_dir: plink_params.output_dir,
        region: base_params.regions,
        overwrite: plink_params.overwrite = False,
        n_snps: Optional[base_params.n_snps] = None,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        size: ld_params.size = ld_params.size_default,
        step: ld_params.step = ld_params.step_default,
        threshold: ld_params.threshold = ld_params.threshold_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ):
        # Include a compact hash for selection parameters that affect output
        # content but are not suitable to place directly in the filename.
        params_hash, _ = _hash_params(
            dict(
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
                sample_indices=sample_indices,
                site_mask=site_mask,
                site_class=site_class,
                cohort_size=cohort_size,
                min_cohort_size=min_cohort_size,
                max_cohort_size=max_cohort_size,
                random_seed=random_seed,
            )
        )
        params_hash_short = params_hash[:8]

        # Define output file path using key filtering and LD parameters,
        # plus a short hash of selection parameters.
        plink_file_path = (
            f"{output_dir}/{region}.ld_pruned"
            f".{n_snps}.{min_minor_ac}.{max_missing_an}"
            f".{thin_offset}.{size}.{step}.{threshold}.{params_hash_short}"
        )

        bed_file_path = f"{plink_file_path}.bed"

        # Check to see if file exists; if overwrite is False, return existing.
        if os.path.exists(bed_file_path) and not overwrite:
            return plink_file_path

        # Get LD-pruned biallelic SNP calls.
        ds_pruned = self.biallelic_snps_ld_pruned(  # type: ignore[attr-defined]
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
            site_mask=site_mask,
            site_class=site_class,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            size=size,
            step=step,
            threshold=threshold,
            inline_array=inline_array,
            chunks=chunks,
        )

        self._write_plink(ds_pruned, bed_file_path)

        return plink_file_path
