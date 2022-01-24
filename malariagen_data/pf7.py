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
        extended_calldata_variables=None,
        extended_variant_fields=None,
        **kwargs,
    ):

        # setup filesystem
        self._fs, self._path = init_filesystem(url, **kwargs)
        self.CONF = self._load_config(data_config)

        # setup caches
        self._cache_sample_metadata = None
        self._cache_zarr = None
        if extended_calldata_variables:
            self.extended_calldata_variables = extended_calldata_variables
        else:
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
        if extended_variant_fields:
            self.extended_variant_fields = extended_variant_fields
        else:
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

    def variant_calls(self, extended=False, inline_array=True, chunks="native"):
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
        # variant_position
        pos_z = root["variants/POS"]
        variant_position = da_from_zarr(pos_z, inline_array=inline_array, chunks=chunks)
        coords["variant_position"] = [DIM_VARIANT], variant_position

        # variant_chrom
        chrom_z = root["variants/CHROM"]
        variant_chrom = da_from_zarr(chrom_z, inline_array=inline_array, chunks=chunks)
        coords["variant_chrom"] = [DIM_VARIANT], variant_chrom

        # variant_allele
        ref_z = root["variants/REF"]
        alt_z = root["variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # variant_filter_pass
        fp_z = root["variants/FILTER_PASS"]
        fp = da_from_zarr(fp_z, inline_array=inline_array, chunks=chunks)
        data_vars["variant_filter_pass"] = [DIM_VARIANT], fp

        # variant_is_snp
        is_snp_z = root["variants/is_snp"]
        is_snp = da_from_zarr(is_snp_z, inline_array=inline_array, chunks=chunks)
        data_vars["variant_is_snp"] = [DIM_VARIANT], is_snp

        # variant_numalt
        numalt_z = root["variants/numalt"]
        numalt = da_from_zarr(numalt_z, inline_array=inline_array, chunks=chunks)
        data_vars["variant_numalt"] = [DIM_VARIANT], numalt

        # variant_CDS
        cds_z = root["variants/CDS"]
        cds = da_from_zarr(cds_z, inline_array=inline_array, chunks=chunks)
        data_vars["variant_CDS"] = [DIM_VARIANT], cds

        # call arrays
        gt_z = root["calldata/GT"]
        call_genotype = da_from_zarr(gt_z, inline_array=inline_array, chunks=chunks)
        ad_z = root["calldata/AD"]
        call_ad = da_from_zarr(ad_z, inline_array=inline_array, chunks=chunks)
        data_vars["call_genotype"] = (
            [DIM_VARIANT, DIM_SAMPLE, DIM_PLOIDY],
            call_genotype,
        )
        data_vars["call_AD"] = ([DIM_VARIANT, DIM_SAMPLE, DIM_ALLELE], call_ad)

        # sample arrays
        z = root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        coords["sample_id"] = [DIM_SAMPLE], sample_id

        if extended:
            for var_name in self.extended_calldata_variables:
                z = root[f"calldata/{var_name}"]
                var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
                data_vars[f"call_{var_name}"] = (
                    self.extended_calldata_variables[var_name],
                    var,
                )

            for var_name in self.extended_variant_fields:
                field_z = root[f"variants/{var_name}"]
                field = da_from_zarr(field_z, inline_array=inline_array, chunks=chunks)
                data_vars[f"variant_{var_name}"] = (
                    self.extended_variant_fields[var_name],
                    field,
                )

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords)

        return ds
