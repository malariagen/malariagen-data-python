import os
import unittest
from unittest.mock import mock_open, patch

import fsspec
import gcsfs
import pandas as pd

from malariagen_data.pf7 import Pf7


class TestPf7(unittest.TestCase):
    def setUp(self):
        self.url_starts_with_gs = "gs://test_url"
        self.url_starts_with_gcs = "gcs://test_url"
        self.url_includes_gs = "some_prefix/gs://test_url"
        self.url_includes_gcs = "some_prefix/gcs://test_url"
        self.url_not_cloud = "/local/path"
        self.url_trailing_slash = "/local/path/"
        self.data_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_config_path = os.path.join(self.data_dir, "test_pf7_config.json")
        self.test_data_path = os.path.join(self.data_dir, "pf7_test_data")
        self.test_pf7_class = Pf7(
            self.test_data_path, data_config="test_pf7_config.json"
        )
        self.test_metadata_path = os.path.join(
            self.test_data_path, "metadata/test_metadata.txt"
        )
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

    @patch(
        "malariagen_data.pf7.Pf7._process_url_with_fsspec", return_value=["fs", "path"]
    )
    @patch("malariagen_data.pf7.Pf7._load_data_structure", return_value="config")
    def test_set_cloud_access(self, mock_process_url, mock_load_data_structure):
        pf7 = Pf7("/fake_path")
        self.assertEqual(
            pf7._set_cloud_access(self.url_starts_with_gs, {})["token"], "anon"
        )
        self.assertEqual(
            pf7._set_cloud_access(self.url_starts_with_gcs, {})["token"], "anon"
        )
        self.assertEqual(
            pf7._set_cloud_access(self.url_includes_gs, {})["gs"]["token"], "anon"
        )
        self.assertEqual(
            pf7._set_cloud_access(self.url_includes_gcs, {})["gcs"]["token"], "anon"
        )
        self.assertEqual(pf7._set_cloud_access(self.url_not_cloud, {}), {})

    @patch("malariagen_data.pf7.Pf7._set_cloud_access", return_value={})
    @patch("malariagen_data.pf7.Pf7._load_data_structure", return_value="config")
    def test_process_url_with_fsspec_returns_correct_file_system(
        self, mock_set_cloud_access, mock_load_data_structure
    ):
        pf7 = Pf7("/fake_path")

        fs, path = pf7._process_url_with_fsspec(self.url_starts_with_gs)
        self.assertIsInstance(
            fs,
            gcsfs.core.GCSFileSystem,
        )
        self.assertEqual(path, "test_url")

        fs, path = pf7._process_url_with_fsspec(self.url_not_cloud)
        self.assertIsInstance(
            fs,
            fsspec.implementations.local.LocalFileSystem,
        )
        self.assertEqual(path, "/local/path")

        fs, path = pf7._process_url_with_fsspec(self.url_trailing_slash)
        self.assertIsInstance(
            fs,
            fsspec.implementations.local.LocalFileSystem,
        )
        self.assertEqual(path, "/local/path")

    @patch("malariagen_data.pf7.Pf7._set_cloud_access", return_value={})
    @patch(
        "malariagen_data.pf7.Pf7._process_url_with_fsspec", return_value=["fs", "path"]
    )
    def test_load_data_structure(self, _process_url_with_fsspec, mock_set_cloud_access):
        pf7 = Pf7("/fake_path", data_config=self.test_config_path)
        config = pf7._load_data_structure(self.test_config_path)
        self.assertEqual(
            config,
            {
                "metadata_path": "metadata/test_metadata.txt",
                "zarr_path": "pf7.zarr/",
            },
        )

    def test_read_general_metadata_returns_correct_data(self):
        df = self.test_pf7_class._read_general_metadata()
        pd.testing.assert_frame_equal(df, self.test_metadata_df)

    @patch("builtins.open", new_callable=mock_open)
    @patch("malariagen_data.pf7.pd.read_csv", return_value=pd.DataFrame())
    def test_read_general_metadata_calls_open_correctly(self, mock_read_csv, mock_open):
        self.test_pf7_class._read_general_metadata()
        mock_open.assert_called_once_with(self.test_metadata_path, mode="rb")
        mock_read_csv.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("malariagen_data.pf7.pd.read_csv", return_value=pd.DataFrame())
    def test_read_general_metadata_with_cache(self, mock_read_csv, mock_open):
        self.test_pf7_class._read_general_metadata()
        self.test_pf7_class._read_general_metadata()
        mock_open.assert_called_once()
        mock_read_csv.assert_called_once()

    @patch("malariagen_data.pf7.Pf7._read_general_metadata")
    def test_sample_metadata(self, mock_read_general_metadata):
        self.test_pf7_class.sample_metadata()
        mock_read_general_metadata.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
