import os
import unittest
from unittest.mock import mock_open, patch

import dask.array as da
import numpy as np
import pandas as pd
import zarr

from malariagen_data.pf7 import DIM_ALT_ALLELE, DIM_GENOTYPES, DIM_STATISTICS, Pf7
from malariagen_data.util import DIM_PLOIDY, DIM_SAMPLE, DIM_VARIANT


class TestPf7(unittest.TestCase):
    def setUp(self):
        self.url_starts_with_gs = "gs://test_url"
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_config_path = os.path.join(self.working_dir, "test_pf7_config.json")
        self.test_data_path = os.path.join(self.working_dir, "pf7_test_data")
        self.test_pf7_class = Pf7(
            self.test_data_path, data_config=self.test_config_path
        )
        self.test_zarr_path = os.path.join(self.test_data_path, "pf7.zarr/")

        # Datasets
        self.config_content = {
            "metadata_path": "metadata/test_metadata.txt",
            "zarr_path": "pf7.zarr/",
        }
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

    @patch("malariagen_data.pf7.init_filesystem", return_value=["fs", "path"])
    def test_setup_with_config_none(self, mock_init_filesystem):
        with patch(
            "malariagen_data.pf7.Pf7._load_config",
            return_value=self.config_content,
        ) as mock_load_config:
            pf7 = Pf7(self.url_starts_with_gs)
        self.assertEqual(pf7._cache_sample_metadata, None)
        self.assertEqual(pf7._cache_zarr, None)
        self.assertEqual(
            pf7.extended_calldata_variables, self.test_extended_calldata_variables
        )
        self.assertEqual(pf7.extended_variant_fields, self.test_extended_variant_fields)
        mock_load_config.assert_called_once_with(None)
        mock_init_filesystem.assert_called_once_with(self.url_starts_with_gs)

    @patch("malariagen_data.pf7.init_filesystem", return_value=["fs", "path"])
    def test_setup_with_config_set(self, mock_init_filesystem):
        with patch(
            "malariagen_data.pf7.Pf7._load_config",
            return_value=self.config_content,
        ) as mock_load_config:
            pf7 = Pf7(self.url_starts_with_gs, data_config=self.test_config_path)
        self.assertEqual(pf7._cache_sample_metadata, None)
        self.assertEqual(pf7._cache_zarr, None)
        self.assertEqual(
            pf7.extended_calldata_variables, self.test_extended_calldata_variables
        )
        self.assertEqual(pf7.extended_variant_fields, self.test_extended_variant_fields)
        mock_load_config.assert_called_once_with(self.test_config_path)
        mock_init_filesystem.assert_called_once_with(self.url_starts_with_gs)

    def test_load_config(self):
        config = self.test_pf7_class._load_config(self.test_config_path)
        self.assertEqual(
            config,
            {
                "metadata_path": "metadata/test_metadata.txt",
                "zarr_path": "pf7.zarr/",
            },
        )

    def test_sample_metadata_returns_extpected_df(self):
        metadata_df = self.test_pf7_class.sample_metadata()
        pd.testing.assert_frame_equal(metadata_df, self.test_metadata_df)

    def test_read_general_metadata_returns_expected_df_with_cache_set(
        self,
    ):
        self.test_pf7_class.sample_metadata()
        metadata_df = self.test_pf7_class.sample_metadata()
        pd.testing.assert_frame_equal(metadata_df, self.test_metadata_df)

    @patch("builtins.open", new_callable=mock_open)
    @patch("malariagen_data.pf7.pd.read_csv", return_value=pd.DataFrame())
    def test_read_general_metadata_uses_cache_when_set(self, mock_read_csv, mock_open):
        self.test_pf7_class.sample_metadata()
        self.test_pf7_class.sample_metadata()
        mock_open.assert_called_once()
        mock_read_csv.assert_called_once()

    @patch("malariagen_data.pf7.init_zarr_store", return_value="Safe store object")
    @patch("malariagen_data.pf7.zarr.open")
    def test_open_zarr_calls_functions_correctly(self, mock_zarr, mock_safestore):
        with patch(
            "malariagen_data.pf7.init_filesystem",
            return_value=["fs", self.test_data_path],
        ):
            pf7_mock_fs = Pf7(self.test_data_path, data_config=self.test_config_path)
        pf7_mock_fs.open_zarr()
        mock_safestore.assert_called_once_with(fs="fs", path=self.test_zarr_path)
        mock_zarr.assert_called_once_with(store="Safe store object")

    @patch("malariagen_data.pf7.init_zarr_store", return_value="Safe store object")
    @patch("malariagen_data.pf7.zarr.open")
    def test_open_zarr_uses_cache(self, mock_zarr, mock_safestore):
        with patch(
            "malariagen_data.pf7.init_filesystem",
            return_value=["fs", self.test_data_path],
        ):
            pf7_mock_fs = Pf7(self.test_data_path, data_config=self.test_config_path)
        pf7_mock_fs.open_zarr()
        pf7_mock_fs.open_zarr()
        mock_safestore.assert_called_once_with(fs="fs", path=self.test_zarr_path)
        mock_zarr.assert_called_once_with(store="Safe store object")

    def test_subset_extended_dictionary(self):
        (
            actual_variants_dictionary,
            actual_calldata_dictionary,
        ) = self.test_pf7_class.subset_extended_dictionary(
            ["DP", "GQ", "AC", "AF", "ANN_Feature_Type"]
        )
        expected_calldata_dictionary = {
            "DP": [DIM_VARIANT, DIM_SAMPLE],
            "GQ": [DIM_VARIANT, DIM_SAMPLE],
        }
        expected_variants_dictionary = {
            "DP": [DIM_VARIANT],
            "AC": [DIM_VARIANT, DIM_ALT_ALLELE],
            "AF": [DIM_VARIANT, DIM_ALT_ALLELE],
            "ANN_Feature_Type": [DIM_VARIANT, DIM_ALT_ALLELE],
        }
        self.assertEqual(actual_variants_dictionary, expected_variants_dictionary)
        self.assertEqual(actual_calldata_dictionary, expected_calldata_dictionary)

    def test_subset_extended_dictionary_raises_error(self):
        with self.assertRaises(ValueError):
            self.test_pf7_class.subset_extended_dictionary(["DP", "AC", "BAD_VARIABLE"])

    def test_add_coordinates(self):
        var_names_for_outputs = {
            "POS": "position",
            "CHROM": "chrom",
            "FILTER_PASS": "filter_pass",
        }
        actual_coordinates = self.test_pf7_class.add_coordinates(
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
        actual_vars = self.test_pf7_class.add_data_vars(
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
        actual_extended = self.test_pf7_class.add_extended_data(
            self.test_zarr_root, True, "native", self.test_extended, {}
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

    def test_add_extended_data_raises_error(self):
        with self.assertRaises(ValueError):
            self.test_pf7_class.add_extended_data(
                self.test_zarr_root, True, "native", {"key": "value"}, {}
            )

    @patch("malariagen_data.pf7.Pf7.open_zarr")
    def test_variant_calls_default(self, mock_open_zarr):
        mock_open_zarr.return_value = self.test_zarr_root
        ds = self.test_pf7_class.variant_calls(inline_array=True, chunks="native")
        mock_open_zarr.assert_called_once_with()
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

    @patch("malariagen_data.pf7.Pf7.open_zarr")
    def test_variant_calls_extended(self, mock_open_zarr):
        mock_open_zarr.return_value = self.test_zarr_root
        test_pf7_class_extended = Pf7(
            self.test_data_path,
            data_config=self.test_config_path,
        )
        ds = test_pf7_class_extended.variant_calls(
            extended=self.test_extended,
            inline_array=True,
            chunks="native",
        )
        mock_open_zarr.assert_called_once_with()
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
