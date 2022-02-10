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
    """Provides access to data from the Pf7 release.
    Parameters
    ----------
    url : str, optional
        Base path to data. Default uses Google Cloud Storage "gs://pf7_release/",
        or specify a local path on your file system if data have been downloaded.
    data_config : str, optional
        Path to config for structure of Pf7 data resource. Defaults to config included
        with the malariagen_data package.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> pf7 = malariagen_data.Pf7()

    Access data downloaded to a local file system:

        >>> pf7 = malariagen_data.Pf7("/local/path/to/pf7_release/")

    """

    def __init__(
        self,
        url=None,
        data_config=None,
        **kwargs,
    ):

        # setup filesystem
        self.CONF = self._load_config(data_config)
        if not url:
            url = self.CONF["default_url"]
        self._fs, self._path = init_filesystem(url, **kwargs)

        # setup caches
        self._cache_sample_metadata = None
        self._cache_variant_calls_zarr = None

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
        """Load the config for data structure on the cloud into json format."""
        if not data_config:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(working_dir, "pf7_config.json")
        with open(data_config) as pf7_json_conf:
            config_content = json.load(pf7_json_conf)
        return config_content

    def sample_metadata(self):
        """Access sample metadata and return as pandas dataframe.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata on the samples that were sequenced as part of this resource.
            Includes the time and place of collection, quality metrics, and accesion numbers.
            One row per sample.

        Example
        -------
        Access metadata as pandas dataframe:

            >>> pf7.sample_metadata()

        """
        if self._cache_sample_metadata is None:
            path = os.path.join(self._path, self.CONF["metadata_path"])
            with self._fs.open(path) as f:
                self._cache_sample_metadata = pd.read_csv(f, sep="\t", na_values="")
        return self._cache_sample_metadata

    def open_variant_calls_zarr(self):
        """Open variant calls zarr.

        Returns
        -------
        root : zarr.hierarchy.Group
            Root of zarr containing information on variant calls.

        """
        if self._cache_variant_calls_zarr is None:
            path = os.path.join(self._path, self.CONF["variant_calls_zarr_path"])
            store = init_zarr_store(fs=self._fs, path=path)
            self._cache_variant_calls_zarr = zarr.open_consolidated(store=store)
        return self._cache_variant_calls_zarr

    def _add_coordinates(self, root, inline_array, chunks, var_names_for_outputs):
        """Add coordinate variables in zarr to dictionary"""
        # coordinates
        coords = dict()
        for var_name in "POS", "CHROM":
            z = root[f"variants/{var_name}"]
            var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            coords[f"variant_{var_names_for_outputs[var_name]}"] = [DIM_VARIANT], var

        z = root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        coords["sample_id"] = [DIM_SAMPLE], sample_id
        return coords

    def _add_data_vars(self, root, inline_array, chunks, var_names_for_outputs):
        """Add default set of variables from zarr to dictionary"""
        data_vars = dict()

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

        return data_vars

    def _add_extended_data(self, root, inline_array, chunks, data_vars):
        """Add all variables not included in default set to dictionary"""

        subset_extended_variants = self.extended_variant_fields
        subset_extended_calldata = self.extended_calldata_variables

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
        return data_vars

    def variant_calls(self, extended=False, inline_array=True, chunks="native"):
        """Access variant sites, site filters and genotype calls.

        Parameters
        ----------
        extended : bool, optional
            If False only the default variables are returned. If True all variables from the zarr are returned. Defaults to False.
        inline_array : bool, optional
            Passed through to dask.array.from_array(). Defaults to True.
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'. Defaults to "native".

        Returns
        -------
        ds : xarray.Dataset
            Dataset containing either default or extended variables from the variant calls Zarr.

        Examples
        --------
        Access core set of variables for variant calls (default):

            >>> pf7.variant_calls()

        Access extended set of variables for variant calls:

            >>> pf7.variant_calls(extended=True)

        """
        # setup
        root = self.open_variant_calls_zarr()
        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }

        # Add default data
        coords = self._add_coordinates(
            root, inline_array, chunks, var_names_for_outputs
        )
        data_vars = self._add_data_vars(
            root, inline_array, chunks, var_names_for_outputs
        )

        # Add extended data
        if extended:
            data_vars = self._add_extended_data(root, inline_array, chunks, data_vars)

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords)

        return ds
