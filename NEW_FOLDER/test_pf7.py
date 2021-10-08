import os
import unittest
from unittest.mock import call, mock_open, patch

import dask.array as da
import fsspec
import gcsfs
import pandas as pd
import zarr

from malariagen_data.pf7 import Pf7


class TestPf7(unittest.TestCase):
    def setUp(self):
        self.url_starts_with_gs = "gs://test_url"
        self.url_starts_with_gcs = "gcs://test_url"
        self.url_includes_gs = "some_prefix/gs://test_url"
        self.url_includes_gcs = "some_prefix/gcs://test_url"
        self.url_not_cloud = "/local/path"
        self.url_trailing_slash = "/local/path/"
        self.working_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_config_path = os.path.join(self.working_dir, "test_pf7_config.json")
        self.test_data_path = os.path.join(self.working_dir, "pf7_test_data")
        self.test_pf7_class = Pf7(
            self.test_data_path, data_config=self.test_config_path
        )
        self.test_metadata_path = os.path.join(
            self.test_data_path, "metadata/test_metadata.txt"
        )
        self.test_zarr_path = os.path.join(self.test_data_path, "pf7.zarr/")
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
        variants.create_group("POS")
        variants.create_group("CHROM")
        variants.create_group("FILTER_PASS")
        variants = self.test_zarr_root.create_group("calldata")
        variants.create_group("GT")
        variants.create_group("AD")
        self.url_real_data = "gs://pf7_staging/"
        self.pf7_real_data = Pf7(self.url_real_data)

    @patch(
        "malariagen_data.pf7.Pf7._process_url_with_fsspec", return_value=["fs", "path"]
    )
    @patch("malariagen_data.pf7.Pf7._load_data_structure", return_value="config")
    @patch("malariagen_data.pf7.Pf7._set_cloud_access", return_value={})
    def test_setup_with_config_none(
        self, mock_cloud_access, mock_load_data, mock_process_url
    ):
        pf7 = Pf7(self.url_starts_with_gs)
        self.assertEqual(pf7.kwargs, {})
        self.assertEqual(pf7._cache_general_metadata, None)
        self.assertEqual(pf7._cache_zarr, None)
        mock_cloud_access.assert_called_once_with(self.url_starts_with_gs, {})
        mock_load_data.assert_called_once_with(
            os.path.join(
                os.path.dirname(self.working_dir), "malariagen_data/pf7_config.json"
            )
        )
        mock_process_url.assert_called_once_with(self.url_starts_with_gs)

    @patch(
        "malariagen_data.pf7.Pf7._process_url_with_fsspec", return_value=["fs", "path"]
    )
    @patch("malariagen_data.pf7.Pf7._load_data_structure", return_value="config")
    @patch("malariagen_data.pf7.Pf7._set_cloud_access", return_value={})
    def test_setup_with_config_set(
        self, mock_cloud_access, mock_load_data, mock_process_url
    ):
        pf7 = Pf7(self.url_starts_with_gs, data_config=self.test_data_path)
        self.assertEqual(pf7.kwargs, {})
        self.assertEqual(pf7._cache_general_metadata, None)
        self.assertEqual(pf7._cache_zarr, None)
        mock_cloud_access.assert_called_once_with(self.url_starts_with_gs, {})
        mock_load_data.assert_called_once_with(self.test_data_path)
        mock_process_url.assert_called_once_with(self.url_starts_with_gs)

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
    def test_read_general_metadata_with_cache_set(self, mock_read_csv, mock_open):
        self.test_pf7_class._read_general_metadata()
        self.test_pf7_class._read_general_metadata()
        mock_open.assert_called_once()
        mock_read_csv.assert_called_once()

    @patch("malariagen_data.pf7.Pf7._read_general_metadata")
    def test_sample_metadata_calls_read(self, mock_read_general_metadata):
        self.test_pf7_class.sample_metadata()
        mock_read_general_metadata.assert_called_once_with()

    @patch("malariagen_data.pf7.FSMap", return_value="mocked_fsmap")
    @patch("malariagen_data.pf7.SafeStore", return_value="Safe store object")
    @patch("malariagen_data.pf7.zarr.open")
    def test_open_zarr_calls_functions_correctly(
        self, mock_zarr, mock_safestore, mock_fsmap
    ):
        with patch(
            "malariagen_data.pf7.Pf7._process_url_with_fsspec",
            return_value=["fs", self.test_data_path],
        ):
            pf7_mock_fs = Pf7(self.test_data_path, data_config=self.test_config_path)
        pf7_mock_fs.open_zarr()

        mock_fsmap.assert_called_once_with(
            root=self.test_zarr_path, fs="fs", check=False, create=False
        )
        mock_safestore.assert_called_once_with("mocked_fsmap")
        mock_zarr.assert_called_once_with(store="Safe store object")

    @patch("malariagen_data.pf7.Pf7.open_zarr")
    @patch("malariagen_data.pf7.from_zarr")
    def test_variants_calls_correctly_with_field_as_none(
        self, mock_from_zarr, mock_open_zarr
    ):
        mock_open_zarr.return_value = self.test_zarr_root
        self.test_pf7_class.variants()
        mock_open_zarr.assert_has_calls([(), ()])
        mock_from_zarr.assert_has_calls(
            [
                call(
                    self.test_zarr_root["variants"]["CHROM"],
                    chunks="native",
                    inline_array=True,
                ),
                call(
                    self.test_zarr_root["variants"]["POS"],
                    chunks="native",
                    inline_array=True,
                ),
            ]
        )

    @patch("malariagen_data.pf7.Pf7.open_zarr")
    @patch("malariagen_data.pf7.from_zarr")
    def test_variants_calls_correctly_with_field_set(
        self, mock_from_zarr, mock_open_zarr
    ):
        mock_open_zarr.return_value = self.test_zarr_root
        self.test_pf7_class.variants(field="FILTER_PASS")
        mock_open_zarr.assert_called_once()
        mock_from_zarr.assert_called_once_with(
            self.test_zarr_root["variants"]["FILTER_PASS"],
            chunks="native",
            inline_array=True,
        )

    @patch("malariagen_data.pf7.Pf7.open_zarr")
    @patch("malariagen_data.pf7.from_zarr")
    def test_calldata_calls_correctly(self, mock_from_zarr, mock_open_zarr):
        mock_open_zarr.return_value = self.test_zarr_root
        self.test_pf7_class.calldata()
        mock_open_zarr.assert_called_once()
        mock_from_zarr.assert_called_once_with(
            self.test_zarr_root["calldata"]["GT"],
            inline_array=True,
            chunks="native",
        )

    @patch("malariagen_data.pf7.Pf7.open_zarr")
    @patch("malariagen_data.pf7.from_zarr")
    def test_calldata_calls_correctly_with_field_set(
        self, mock_from_zarr, mock_open_zarr
    ):
        mock_open_zarr.return_value = self.test_zarr_root
        self.test_pf7_class.calldata(field="AD")
        mock_open_zarr.assert_called_once()
        mock_from_zarr.assert_called_once_with(
            self.test_zarr_root["calldata"]["AD"],
            inline_array=True,
            chunks="native",
        )

    def test_open_zarr_on_real_data(self):
        root = self.pf7_real_data.open_zarr()
        self.assertIsInstance(root, zarr.hierarchy.Group)

    def test_variants_on_real_data(self):
        chrom, pos = self.pf7_real_data.variants()
        self.assertIsInstance(chrom, da.Array)
        self.assertEqual(chrom.ndim, 1)
        self.assertEqual(chrom.dtype, "O")
        self.assertIsInstance(pos, da.Array)
        self.assertEqual(pos.ndim, 1)
        self.assertEqual(pos.dtype, "i4")
        assert pos.shape[0] == chrom.shape[0]

    def test_variants_on_real_data_with_field(self):
        pos = self.pf7_real_data.variants(field="FILTER_PASS")
        self.assertIsInstance(pos, da.Array)
        self.assertEqual(pos.ndim, 1)
        self.assertEqual(pos.dtype, "bool")

    def test_calldata_on_real_data(self):
        gt = self.pf7_real_data.calldata()
        self.assertIsInstance(gt, da.Array)
        self.assertEqual(gt.ndim, 3)
        self.assertEqual(gt.dtype, "int8")


if __name__ == "__main__":
    unittest.main()
