import os
import unittest
from unittest.mock import mock_open, patch

import dask.array as da
import numpy as np
import pandas as pd
import zarr

from malariagen_data.plasmodium import PlasmodiumTools
from malariagen_data.util import DIM_PLOIDY, DIM_SAMPLE, DIM_VARIANT

DIM_ALT_ALLELE = "alt_alleles"
DIM_STATISTICS = "sb_statistics"
DIM_GENOTYPES = "genotypes"


class TestPlasmodiumTools(unittest.TestCase):
    def setUp(self):
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_config_path = os.path.join(
            self.working_dir, "test_plasmodium_config.json"
        )
        self.test_data_path = os.path.join(self.working_dir, "plasmodium_test_data")
        self.test_plasmodium_class = PlasmodiumTools(
            self.test_config_path,
            url=self.test_data_path,
        )

        # Datasets
        self.d = {
            "Sample": ["sample1", "sample2", "sample3", "sample4"],
            "Study": ["study1", "study2", "study3", "study1"],
            "Country": ["Mauritania", "Mauritania", "Ghana", "Mauritania"],
            "Admin level 1": [
                "Hodh el Gharbi",
                "Hodh el Gharbi",
                "Upper East",
                "Hodh el Gharbi",
            ],
            "Country latitude": [20.265, 20.265, 7.966, 20.265],
            "Country longitude": [-10.337, -10.337, -1.2109999999999999, -10.337],
            "Admin level 1 latitude": [1.12, 3.45, 6.78, 20.52],
            "Admin level 1 longitude": [-9.42, -8.73, -9.43, -9.87],
            "Year": [2014, 2014, 2009, 2012],
            "ENA": ["ERR1", "ERR2", "ERR3", "ERR4"],
            "All samples same case": ["sample", "sample", "sample", "sample"],
            "Population": ["AF-W", "AF-W", "AF-W", "AF-W"],
            "% callable": [81.25, 88.36, 86.46, 82.15],
            "QC pass": [True, True, True, True],
            "Exclusion reason": [
                "Analysis_set",
                "Analysis_set",
                "Analysis_set",
                "Analysis_set",
            ],
            "Sample type": ["gDNA", "gDNA", "sWGA", "sWGA"],
            "Sample was in Pf6": [True, True, False, False],
        }
        self.test_metadata_df = pd.DataFrame(data=self.d)
        self.test_zarr_root = zarr.group()
        variants = self.test_zarr_root.create_group("variants")
        variants.create_dataset("POS", data=np.arange(10), chunks=(1,))
        variants.create_dataset("CHROM", data=np.arange(10), chunks=(1,))
        variants.create_dataset("REF", data=np.arange(10), chunks=(4194304,))
        variants.create_dataset(
            "ALT", data=np.arange(60).reshape(10, 6), chunks=(699051,)
        )
        variants.create_dataset("FILTER_PASS", data=np.arange(10), chunks=(1,))
        variants.create_dataset("is_snp", data=np.arange(10), chunks=(1,))
        variants.create_dataset("numalt", data=np.arange(10), chunks=(1,))
        variants.create_dataset("CDS", data=np.arange(10), chunks=(1,))
        variants.create_dataset("AN", data=np.arange(10), chunks=(1,))
        calldata = self.test_zarr_root.create_group("calldata")
        calldata.create_dataset(
            "GT", data=np.arange(100).reshape(10, 5, 2), chunks=(1,)
        )
        calldata.create_dataset(
            "AD", data=np.arange(350).reshape(10, 5, 7), chunks=(1,)
        )
        calldata.create_dataset("GQ", data=np.arange(50).reshape(10, 5), chunks=(1,))
        self.test_zarr_root.create_dataset("samples", data=np.arange(5), chunks=(1,))
        self.test_extended = ["AN", "GQ"]
        self.test_extended_calldata_variables = {
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
        self.test_extended_variant_fields = {
            "AC": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AF": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AN": [DIM_VARIANT],
            "ANN_AA_length": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_AA_pos": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Allele": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Annotation": [DIM_VARIANT, DIM_ALT_ALLELE],
            "RAW_MQandDP": [DIM_VARIANT, DIM_PLOIDY],
        }

    def test_setup_returns_config_correctly(self):
        plasmodium = PlasmodiumTools(self.test_config_path)
        self.assertEqual(
            plasmodium.CONF,
            {
                "default_url": "gs://test_plasmodium_release/",
                "metadata_path": "metadata/test_metadata.txt",
                "variant_calls_zarr_path": "test_plasmodium.zarr/",
                "default_variant_variables": {
                    "FILTER_PASS": ["variants"],
                    "is_snp": ["variants"],
                    "numalt": ["variants"],
                    "CDS": ["variants"],
                },
                "extended_calldata_variables": {
                    "DP": ["variants", "samples"],
                    "GQ": ["variants", "samples"],
                    "MIN_DP": ["variants", "samples"],
                    "PGT": ["variants", "samples"],
                    "PID": ["variants", "samples"],
                    "PS": ["variants", "samples"],
                    "RGQ": ["variants", "samples"],
                    "PL": ["variants", "samples", "genotypes"],
                    "SB": ["variants", "samples", "sb_statistics"],
                },
                "extended_variant_fields": {
                    "AC": ["variants", "alt_alleles"],
                    "AF": ["variants", "alt_alleles"],
                    "AN": ["variants"],
                    "ANN_AA_length": ["variants", "alt_alleles"],
                    "ANN_AA_pos": ["variants", "alt_alleles"],
                    "ANN_Allele": ["variants", "alt_alleles"],
                    "ANN_Annotation": ["variants", "alt_alleles"],
                    "RAW_MQandDP": ["variants", "ploidy"],
                },
            },
        )
        self.assertEqual(plasmodium._path, "test_plasmodium_release")

    @patch("malariagen_data.plasmodium.PlasmodiumTools._load_config")
    def test_setup_overrides_default_url(self, mock_load_config):
        url_starts_with_gs = "gs://test_url"
        plasmodium = PlasmodiumTools(self.test_config_path, url=url_starts_with_gs)
        self.assertEqual(plasmodium._path, "test_url")
        mock_load_config.assert_called_once_with(self.test_config_path)

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    def test_load_config_calls_path(self, mock_load_json, mock_open):
        self.test_plasmodium_class._load_config(self.test_config_path)
        open.assert_called_once_with(self.test_config_path)

    def test_sample_metadata_returns_extpected_df(self):
        metadata_df = self.test_plasmodium_class.open_sample_metadata()
        pd.testing.assert_frame_equal(metadata_df, self.test_metadata_df)

    def test_sample_metadata_returns_expected_df_with_cache_set(
        self,
    ):
        self.test_plasmodium_class.open_sample_metadata()
        metadata_df = self.test_plasmodium_class.open_sample_metadata()
        pd.testing.assert_frame_equal(metadata_df, self.test_metadata_df)

    @patch("builtins.open", new_callable=mock_open)
    @patch("malariagen_data.plasmodium.pd.read_csv", return_value=pd.DataFrame())
    def test_sample_metadata_uses_cache_when_set(self, mock_read_csv, mock_open):
        self.test_plasmodium_class.open_sample_metadata()
        self.test_plasmodium_class.open_sample_metadata()
        mock_open.assert_called_once()
        mock_read_csv.assert_called_once()

    @patch(
        "malariagen_data.plasmodium.init_zarr_store", return_value="Safe store object"
    )
    @patch("malariagen_data.plasmodium.zarr.open_consolidated")
    def test_open_variant_calls_zarr_uses_cache(self, mock_zarr, mock_safestore):
        with patch(
            "malariagen_data.plasmodium.init_filesystem",
            return_value=["fs", self.test_data_path],
        ):
            plas_mock_fs = PlasmodiumTools(
                self.test_config_path, url=self.test_data_path
            )
        plas_mock_fs.open_variant_calls_zarr()
        plas_mock_fs.open_variant_calls_zarr()
        test_zarr_path = os.path.join(self.test_data_path, "test_plasmodium.zarr/")
        mock_safestore.assert_called_once_with(fs="fs", path=test_zarr_path)
        mock_zarr.assert_called_once_with(store="Safe store object")

    def test_add_coordinates(self):
        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }
        actual_coordinates = self.test_plasmodium_class._add_coordinates(
            self.test_zarr_root, True, "native", var_names_for_outputs
        )
        self.assertEqual(
            list(actual_coordinates.keys()),
            ["variant_position", "variant_chrom", "sample_id"],
        )

        values = [actual_coordinates[k] for k in actual_coordinates]
        for value in values:
            assert isinstance(value[1], da.Array)
        dimensions = [i[0] for i in values]
        self.assertEqual(
            dimensions,
            [["variants"], ["variants"], ["samples"]],
        )

    def test_add_data_vars(self):
        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }
        actual_vars = self.test_plasmodium_class._add_default_data_vars(
            self.test_zarr_root, True, "native", var_names_for_outputs
        )
        self.assertEqual(
            list(actual_vars.keys()),
            [
                "variant_allele",
                "variant_filter_pass",
                "variant_is_snp",
                "variant_numalt",
                "variant_CDS",
                "call_genotype",
                "call_AD",
            ],
        )
        values = [actual_vars[k] for k in actual_vars]
        for value in values:
            assert isinstance(value[1], da.Array)
        dimensions = [i[0] for i in values]
        self.assertEqual(
            dimensions,
            [
                ["variants", "alleles"],
                ["variants"],
                ["variants"],
                ["variants"],
                ["variants"],
                ["variants", "samples", "ploidy"],
                ["variants", "samples", "alleles"],
            ],
        )

    def test_add_extended_data(self):
        test_plasmodium_extended = PlasmodiumTools(
            self.test_config_path,
            url=self.test_data_path,
        )
        test_plasmodium_extended.extended_variant_fields = {"AN": [DIM_VARIANT]}
        test_plasmodium_extended.extended_calldata_variables = {
            "GQ": [DIM_VARIANT, DIM_SAMPLE]
        }
        actual_extended = test_plasmodium_extended._add_extended_data(
            self.test_zarr_root, True, "native", {}
        )
        self.assertEqual(
            list(actual_extended.keys()),
            [
                "call_GQ",
                "variant_AN",
            ],
        )

        values = [actual_extended[k] for k in actual_extended]
        for value in values:
            assert isinstance(value[1], da.Array)
        dimensions = [i[0] for i in values]
        self.assertEqual(
            dimensions,
            [
                ["variants", "samples"],
                ["variants"],
            ],
        )

    @patch("malariagen_data.plasmodium.PlasmodiumTools.open_variant_calls_zarr")
    def test_variant_calls_default(self, mock_open_variant_calls_zarr):
        mock_open_variant_calls_zarr.return_value = self.test_zarr_root
        ds = self.test_plasmodium_class.load_variant_calls(
            inline_array=True, chunks="native"
        )
        mock_open_variant_calls_zarr.assert_called_once_with()
        coords = list(ds.coords.keys())
        variables = list(ds.keys())
        self.assertEqual(coords, ["variant_position", "variant_chrom", "sample_id"])
        self.assertEqual(
            variables,
            [
                "variant_allele",
                "variant_filter_pass",
                "variant_is_snp",
                "variant_numalt",
                "variant_CDS",
                "call_genotype",
                "call_AD",
            ],
        )

    @patch("malariagen_data.plasmodium.PlasmodiumTools.open_variant_calls_zarr")
    def test_variant_calls_extended(self, mock_open_variant_calls_zarr):
        mock_open_variant_calls_zarr.return_value = self.test_zarr_root
        test_plasmodium_class_extended = PlasmodiumTools(
            self.test_config_path,
            url=self.test_data_path,
        )
        test_plasmodium_class_extended.extended_variant_fields = {"AN": [DIM_VARIANT]}
        test_plasmodium_class_extended.extended_calldata_variables = {
            "GQ": [DIM_VARIANT, DIM_SAMPLE]
        }
        ds = test_plasmodium_class_extended.load_variant_calls(
            extended=True,
            inline_array=True,
            chunks="native",
        )
        mock_open_variant_calls_zarr.assert_called_once_with()
        coords = list(ds.coords.keys())
        variables = list(ds.keys())
        self.assertEqual(coords, ["variant_position", "variant_chrom", "sample_id"])
        self.assertEqual(
            variables,
            [
                "variant_allele",
                "variant_filter_pass",
                "variant_is_snp",
                "variant_numalt",
                "variant_CDS",
                "call_genotype",
                "call_AD",
                "call_GQ",
                "variant_AN",
            ],
        )


if __name__ == "__main__":
    unittest.main()
