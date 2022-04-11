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


class PlasmodiumDataResource:
    def __init__(
        self,
        data_config,
        url=None,
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

        self.extended_calldata_variables = self.CONF["extended_calldata_variables"]
        self.extended_variant_fields = self.CONF["extended_variant_fields"]

    def _load_config(self, data_config):
        """Load the config for data structure on the cloud into json format."""
        with open(data_config) as json_conf:
            config_content = json.load(json_conf)
        return config_content

    def sample_metadata(self):
        """Access sample metadata and return as pandas dataframe.
        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata on the samples that were sequenced as part of this resource.
            Includes the time and place of collection, quality metrics, and accesion numbers.
            One row per sample.
        """
        if self._cache_sample_metadata is None:
            path = os.path.join(self._path, self.CONF["metadata_path"])
            with self._fs.open(path) as f:
                self._cache_sample_metadata = pd.read_csv(f, sep="\t", na_values="")
        return self._cache_sample_metadata

    def _open_variant_calls_zarr(self):
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
        for var_name in ["POS", "CHROM"]:
            z = root[f"variants/{var_name}"]
            var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            coords[f"variant_{var_names_for_outputs[var_name]}"] = [DIM_VARIANT], var

        z = root["samples"]
        sample_id = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
        coords["sample_id"] = [DIM_SAMPLE], sample_id
        return coords

    def _add_default_data_vars(self, root, inline_array, chunks, var_names_for_outputs):
        """Add default set of variables from zarr to dictionary"""
        data_vars = dict()

        # Add variant_allele as combination of REF and ALT
        ref_z = root["variants/REF"]
        alt_z = root["variants/ALT"]
        ref = da_from_zarr(ref_z, inline_array=inline_array, chunks=chunks)
        alt = da_from_zarr(alt_z, inline_array=inline_array, chunks=chunks)
        variant_allele = da.concatenate([ref[:, None], alt], axis=1)
        data_vars["variant_allele"] = [DIM_VARIANT, DIM_ALLELE], variant_allele

        # other default variant values
        configurable_default_variant_variables = self.CONF["default_variant_variables"]
        for var_name in configurable_default_variant_variables:
            z = root[f"variants/{var_name}"]
            dimension = configurable_default_variant_variables[var_name]
            var = da_from_zarr(z, inline_array=inline_array, chunks=chunks)
            if var_name in var_names_for_outputs.keys():
                var_name = var_names_for_outputs[var_name]
            data_vars[f"variant_{var_name}"] = dimension, var

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
        """
        # setup
        root = self._open_variant_calls_zarr()
        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }

        # Add default data
        coords = self._add_coordinates(
            root, inline_array, chunks, var_names_for_outputs
        )
        data_vars = self._add_default_data_vars(
            root, inline_array, chunks, var_names_for_outputs
        )

        # Add extended data
        if extended:
            data_vars = self._add_extended_data(root, inline_array, chunks, data_vars)

        # create a dataset
        ds = xarray.Dataset(data_vars=data_vars, coords=coords)

        return ds
