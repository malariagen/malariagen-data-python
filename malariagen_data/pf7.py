import json
import os

import dask.array as da
import pandas as pd
import xarray
import zarr

from malariagen_data.util import (
    DIM_ALLELE,
    DIM_PLOIDY,
    DIM_SAMPLE,
    DIM_VARIANT,
    da_from_zarr,
    init_filesystem,
    init_zarr_store,
)

DIM_ALT_ALLELE = "alt_alleles"
DIM_STATISTICS = "sb_statistics"
DIM_GENOTYPES = "genotypes"


class Pf7:
    def __init__(
        self,
        url,
        data_config=None,
        **kwargs,
    ):

        # setup filesystem
        self._fs, self._path = init_filesystem(url, **kwargs)
        self.CONF = self._load_config(data_config)

        # setup caches
        self._cache_sample_metadata = None
        self._cache_zarr = None

        self.extended_calldata_variables = {
            "DP": [DIM_VARIANT, DIM_SAMPLE],
            "GQ": [DIM_VARIANT, DIM_SAMPLE],
            "MIN_DP": [DIM_VARIANT, DIM_SAMPLE],
            "PGT": [DIM_VARIANT, DIM_SAMPLE],
            "PID": [DIM_VARIANT, DIM_SAMPLE],
            "PS": [DIM_VARIANT, DIM_SAMPLE],
            "RGQ": [DIM_VARIANT, DIM_SAMPLE],
            "PL": [DIM_VARIANT, DIM_SAMPLE, DIM_GENOTYPES],
            "SB": [DIM_VARIANT, DIM_SAMPLE, DIM_STATISTICS],
        }
        self.extended_variant_fields = {
            "AC": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AF": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AN": [DIM_VARIANT],
            "ANN_AA_length": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_AA_pos": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Allele": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Annotation": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Annotation_Impact": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_CDS_length": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_CDS_pos": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Distance": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Feature_ID": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Feature_Type": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Gene_ID": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Gene_Name": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_HGVS_c": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_HGVS_p": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Rank": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Transcript_BioType": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_cDNA_length": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_cDNA_pos": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_BaseQRankSum": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_FS": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_InbreedingCoeff": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_MQ": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_MQRankSum": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_QD": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_ReadPosRankSum": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AS_SOR": [DIM_VARIANT, DIM_ALT_ALLELE],
            "BaseQRankSum": [DIM_VARIANT],
            "DP": [DIM_VARIANT],
            "DS": [DIM_VARIANT],
            "END": [DIM_VARIANT],
            "ExcessHet": [DIM_VARIANT],
            "FILTER_Apicoplast": [DIM_VARIANT],
            "FILTER_Centromere": [DIM_VARIANT],
            "FILTER_InternalHypervariable": [DIM_VARIANT],
            "FILTER_LowQual": [DIM_VARIANT],
            "FILTER_Low_VQSLOD": [DIM_VARIANT],
            "FILTER_MissingVQSLOD": [DIM_VARIANT],
            "FILTER_Mitochondrion": [DIM_VARIANT],
            "FILTER_SubtelomericHypervariable": [DIM_VARIANT],
            "FILTER_SubtelomericRepeat": [DIM_VARIANT],
            "FILTER_VQSRTrancheINDEL99.50to99.60": [DIM_VARIANT],
            "FILTER_VQSRTrancheINDEL99.60to99.80": [DIM_VARIANT],
            "FILTER_VQSRTrancheINDEL99.80to99.90": [DIM_VARIANT],
            "FILTER_VQSRTrancheINDEL99.90to99.95": [DIM_VARIANT],
            "FILTER_VQSRTrancheINDEL99.95to100.00+": [DIM_VARIANT],
            "FILTER_VQSRTrancheINDEL99.95to100.00": [DIM_VARIANT],
            "FILTER_VQSRTrancheSNP99.50to99.60": [DIM_VARIANT],
            "FILTER_VQSRTrancheSNP99.60to99.80": [DIM_VARIANT],
            "FILTER_VQSRTrancheSNP99.80to99.90": [DIM_VARIANT],
            "FILTER_VQSRTrancheSNP99.90to99.95": [DIM_VARIANT],
            "FILTER_VQSRTrancheSNP99.95to100.00+": [DIM_VARIANT],
            "FILTER_VQSRTrancheSNP99.95to100.00": [DIM_VARIANT],
            "FS": [DIM_VARIANT],
            "ID": [DIM_VARIANT],
            "InbreedingCoeff": [DIM_VARIANT],
            "LOF": [DIM_VARIANT],
            "MLEAC": [DIM_VARIANT, DIM_ALT_ALLELE],
            "MLEAF": [DIM_VARIANT, DIM_ALT_ALLELE],
            "MQ": [DIM_VARIANT],
            "MQRankSum": [DIM_VARIANT],
            "NEGATIVE_TRAIN_SITE": [DIM_VARIANT],
            "NMD": [DIM_VARIANT],
            "POSITIVE_TRAIN_SITE": [DIM_VARIANT],
            "QD": [DIM_VARIANT],
            "QUAL": [DIM_VARIANT],
            "RAW_MQandDP": [DIM_VARIANT, DIM_PLOIDY],
            "ReadPosRankSum": [DIM_VARIANT],
            "RegionType": [DIM_VARIANT],
            "SOR": [DIM_VARIANT],
            "VQSLOD": [DIM_VARIANT],
            "altlen": [DIM_VARIANT, DIM_ALT_ALLELE],
            "culprit": [DIM_VARIANT],
            "set": [DIM_VARIANT],
        }

    def _load_config(self, data_config):
        if not data_config:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(working_dir, "pf7_config.json")
        with open(data_config) as pf7_json_conf:
            config_content = json.load(pf7_json_conf)
        return config_content

    def sample_metadata(self):
        """Access sample metadata.
        Returns
        -------
        df : pandas.DataFrame
        """
        if self._cache_sample_metadata is None:
            path = os.path.join(self._path, self.CONF["metadata_path"])
            with self._fs.open(path) as f:
                self._cache_sample_metadata = pd.read_csv(f, sep="\t", na_values="")
        return self._cache_sample_metadata

    def open_zarr(self):
        if self._cache_zarr is None:
            path = os.path.join(self._path, self.CONF["zarr_path"])
            store = init_zarr_store(fs=self._fs, path=path)
            """WARNING: Metadata has not been consolidated yet. Using open for now but will eventually switch to opn_consolidated when the .zmetadata file has been created
            """
            self._cache_zarr = zarr.open(store=store)
        return self._cache_zarr

    def subset_extended_dictionary(self, extended_variables):
        """# subset extended variants and calldata dictionaries to only include items in list

        Args:
            extended_variables (list): list of variables to add onto default dataset

        Raises:
            ValueError: If variable isn't in either variants or calldata dictionary raises error as invalid entry

        Returns:
            subset_extended_variants, subset_extended_calldata : dictionaries containing subset of extended values
        """
        subset_extended_variants = {}
        subset_extended_calldata = {}
        for variable in extended_variables:
            if variable in self.extended_variant_fields:
                subset_extended_variants[variable] = self.extended_variant_fields[
                    variable
                ]
            if variable in self.extended_calldata_variables:
                subset_extended_calldata[variable] = self.extended_calldata_variables[
                    variable
                ]
            if (
                variable not in self.extended_calldata_variables
                and variable not in self.extended_variant_fields
            ):
                raise ValueError("{} not found in zarr.".format(variable))
        return subset_extended_variants, subset_extended_calldata

    def variant_calls(self, extended=[], inline_array=True, chunks="native"):
        """Access variant sites, site filters and genotype calls.
        Parameters
        ----------
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.
        Returns
        -------
        ds : xarray.Dataset
        """

        # setup
        coords = dict()
        data_vars = dict()
        root = self.open_zarr()

        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }

        # coordinates
        for var_name in "POS", "CHROM":
            z = root[f"variants/{var_name}"]
            var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            coords[f"variant_{var_names_for_outputs[var_name]}"] = [DIM_VARIANT], var

        z = root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        # variant_allele
        ref_z = root["variants/REF"]
        alt_z = root["variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # other default variant values
        for var_name in ["FILTER_PASS", "is_snp", "numalt", "CDS"]:
            z = root[f"variants/{var_name}"]
            var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            if var_name in var_names_for_outputs.keys():
                var_name = var_names_for_outputs[var_name]
            data_vars[f"variant_{var_name}"] = [DIM_VARIANT], var

        # call arrays
        gt_z = root["calldata/GT"]
        ad_z = root["calldata/AD"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_AD"] = ([DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE], call_ad)

        # pull the extended version
        if extended:
            if extended == "*":
                subset_extended_variants = self.extended_variant_fields
                subset_extended_calldata = self.extended_calldata_variables
            elif isinstance(extended, list):
                (
                    subset_extended_variants,
                    subset_extended_calldata,
                ) = self.subset_extended_dictionary(extended)
            else:
                raise ValueError("Input to extended is invalid.")

            for var_name in subset_extended_calldata:
                z = root[f"calldata/{var_name}"]
                var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
                data_vars[f"call_{var_name}"] = (
                    subset_extended_calldata[var_name],
                    var,
                )

            for var_name in subset_extended_variants:
                z = root[f"variants/{var_name}"]
                field = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
                data_vars[f"variant_{var_name}"] = (
                    subset_extended_variants[var_name],
                    field,
                )

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords)

        return ds
