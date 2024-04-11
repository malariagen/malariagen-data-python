from typing import Optional

import allel  # type: ignore
import numpy as np
import os
import bed_reader
from dask.diagnostics import ProgressBar
# import bokeh.plotting

from .snp_data import AnophelesSnpData
from . import base_params


# So far all of this has been copied from Sanjay's g123 notebook as it loads snp data with some optional filters.
class PlinkConverter(
    # ADMIXTURE is the main reason I want to convert malaariagen SNP data to PLINK format, and it uses genotype data.
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

    def _create_plink_outfile(
        self,
        *,
        results_dir,
        region,
        n_snps,
        min_minor_ac,
        thin_offset,
        max_missing_an,
    ):
        return f"{results_dir}/{region}.{n_snps}.{min_minor_ac}.{thin_offset}.{max_missing_an}"

    def _biallelic_snps_to_plink(
        # _ means internal function
        # loads biallelic diplotypes and selects segregating sites among them, converts to plink
        self,
        *,
        results_dir,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_query,
        sample_indices,
        site_mask,
        min_minor_ac,
        max_missing_an,
        random_seed,
        inline_array,
        chunks,
    ):
        # first define output file
        plink_file_path = self._create_plink_outfile(
            results_dir=results_dir,
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
        )

        bed_file_path = f"{plink_file_path}.bed"
        if os.path.exists(bed_file_path):
            return plink_file_path

        # get snps
        ds_snps = self.biallelic_snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
            site_mask=site_mask,
            # min_minor_ac=min_minor_ac,
            # max_missing_an=max_missing_an,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
            n_snps=n_snps,
        )

        # filter snps on segregating sites
        with self._spinner("Subsetting to segregating sites"):
            gt = ds_snps["call_genotype"].data.compute()
            print("count alleles")
            with ProgressBar():
                ac = allel.GenotypeArray(gt).count_alleles(max_allele=3)
                print("ascertain segregating  sites")
                n_chroms = ds_snps.dims["samples"] * 2
                an_called = ac.sum(axis=1)
                an_missing = n_chroms - an_called
                min_ref_ac = min_minor_ac
                max_ref_ac = n_chroms - min_minor_ac
                # here we choose segregating sites
                loc_sites = (  # put in is_biallelic
                    (ac[:, 0] >= min_ref_ac)
                    & (ac[:, 0] <= max_ref_ac)
                    & (an_missing <= max_missing_an)
                )
                print(f"ascertained {np.count_nonzero(loc_sites):,} sites")

        # print("thin sites")
        # ix_sites = np.nonzero(loc_sites)[0]
        # thin_step = max(ix_sites.shape[0] // n_snps, 1)
        # ix_sites_thinned = ix_sites[thin_offset::thin_step]
        # print(f"thinned to {np.count_nonzero(ix_sites_thinned):,} sites")

        # set up dataset with required vars for plink conversion
        print("Set up dataset")
        ds_snps_asc = (
            ds_snps[
                [
                    "variant_contig",
                    "variant_position",
                    "variant_allele",
                    "sample_id",
                    "call_genotype",
                ]
            ].isel(alleles=slice(0, 2))
            # .sel(variants=ix_sites_thinned)
        )

        # compute gt ref counts
        with self._spinner("Computing genotype ref counts"):
            gt_asc = ds_snps_asc["call_genotype"].data.compute()
            gn_ref = allel.GenotypeDaskArray(gt_asc).to_n_ref(fill=-127)
            with ProgressBar():
                gn_ref = gn_ref.compute()

        print("Ensure genotypes vary")
        loc_var = np.any(gn_ref != gn_ref[:, 0, np.newaxis], axis=1)
        print(f"final no. variants {np.count_nonzero(loc_var)}")

        print("Load final data")
        with ProgressBar():
            ds_snps_final = ds_snps_asc[
                ["variant_contig", "variant_position", "variant_allele", "sample_id"]
            ].isel(variants=loc_var)

        # init vars for input to bed reader
        gn_ref_final = gn_ref[loc_var]
        val = gn_ref_final.T
        alleles = ds_snps_final["variant_allele"].values
        properties = {
            "iid": ds_snps_final["sample_id"].values,
            "chromosome": ds_snps_final["variant_contig"].values,
            "bp_position": ds_snps_final["variant_position"].values,
            "allele_1": alleles[:, 0],
            "allele_2": alleles[:, 1],
        }

        print(f"write plink files to {plink_file_path}")
        bed_reader.to_bed(
            filepath=bed_file_path,
            val=val,
            properties=properties,
            count_A1=True,
        )

        return plink_file_path

    def biallelic_snps_to_plink(
        self,
        results_dir,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = None,
        min_minor_ac: Optional[base_params.min_minor_ac] = 0,
        max_missing_an: Optional[base_params.max_missing_an] = 0,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ):
        params = dict(
            results_dir=results_dir,
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            random_seed=random_seed,
        )
        # for the sake of getting this going
        # and also because i expect sanjay and ali will rework most of this, i will forgo the type checks until later

        filepath = self._biallelic_snps_to_plink(
            inline_array=inline_array, chunks=chunks, **params
        )
        return filepath
