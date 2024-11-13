import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray

from malariagen_data.pf8 import Pf8


def setup_pf8(url="simplecache::gs://pf8-release/", **storage_kwargs):
    if url.startswith("simplecache::"):
        storage_kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Pf8(url, **storage_kwargs)


@pytest.mark.parametrize(
    "url",
    [
        "gs://pf8-release/",
        "gcs://pf8-release/",
        "gs://pf8-release",
        "gcs://pf8-release",
        "simplecache::gs://pf8-release/",
        "simplecache::gcs://pf8-release/",
    ],
)
def test_sample_metadata(url):
    pf8 = setup_pf8(url)
    df_samples = pf8.sample_metadata()

    expected_cols = (
        "Sample",
        "Study",
        "Country",
        "Admin level 1",
        "Country latitude",
        "Country longitude",
        "Admin level 1 latitude",
        "Admin level 1 longitude",
        "Year",
        "ENA",
        "All samples same case",
        "Population",
        "% callable",
        "QC pass",
        "Exclusion reason",
        "Sample type",
        "Sample was in Pf7",
    )

    assert tuple(df_samples.columns) == expected_cols
    expected_len = 33325
    assert len(df_samples) == expected_len


@pytest.mark.parametrize("extended", [True, False])
def test_variant_calls(extended):
    pf8 = setup_pf8()

    ds = pf8.variant_calls(extended=extended)
    assert isinstance(ds, xarray.Dataset)

    # check fields
    if extended:
        expected_data_vars = {
            "variant_allele",
            "variant_filter_pass",
            "variant_is_snp",
            "variant_numalt",
            "variant_CDS",
            "call_genotype",
            "call_AD",
            "call_DP",
            "call_GQ",
            "call_MIN_DP",
            "call_PGT",
            "call_PID",
            "call_PS",
            "call_RGQ",
            "call_PL",
            "call_SB",
            "variant_AC",
            "variant_AF",
            "variant_AN",
            "variant_ANN_AA_length",
            "variant_ANN_AA_pos",
            "variant_ANN_Allele",
            "variant_ANN_Annotation",
            "variant_ANN_Annotation_Impact",
            "variant_ANN_CDS_length",
            "variant_ANN_CDS_pos",
            "variant_ANN_Distance",
            "variant_ANN_Feature_ID",
            "variant_ANN_Feature_Type",
            "variant_ANN_Gene_ID",
            "variant_ANN_Gene_Name",
            "variant_ANN_HGVS_c",
            "variant_ANN_HGVS_p",
            "variant_ANN_Rank",
            "variant_ANN_Transcript_BioType",
            "variant_ANN_cDNA_length",
            "variant_ANN_cDNA_pos",
            "variant_AS_BaseQRankSum",
            "variant_AS_FS",
            "variant_AS_InbreedingCoeff",
            "variant_AS_MQ",
            "variant_AS_MQRankSum",
            "variant_AS_QD",
            "variant_AS_ReadPosRankSum",
            "variant_AS_SOR",
            "variant_BaseQRankSum",
            "variant_DP",
            "variant_DS",
            "variant_END",
            "variant_ExcessHet",
            "variant_FILTER_Apicoplast",
            "variant_FILTER_Centromere",
            "variant_FILTER_InternalHypervariable",
            "variant_FILTER_LowQual",
            "variant_FILTER_Low_VQSLOD",
            "variant_FILTER_Mitochondrion",
            "variant_FILTER_SubtelomericHypervariable",
            "variant_FILTER_SubtelomericRepeat",
            "variant_FILTER_VQSRTrancheINDEL99.50to99.60",
            "variant_FILTER_VQSRTrancheINDEL99.60to99.80",
            "variant_FILTER_VQSRTrancheINDEL99.80to99.90",
            "variant_FILTER_VQSRTrancheINDEL99.90to99.95",
            "variant_FILTER_VQSRTrancheINDEL99.95to100.00+",
            "variant_FILTER_VQSRTrancheINDEL99.95to100.00",
            "variant_FILTER_VQSRTrancheSNP99.50to99.60",
            "variant_FILTER_VQSRTrancheSNP99.60to99.80",
            "variant_FILTER_VQSRTrancheSNP99.80to99.90",
            "variant_FILTER_VQSRTrancheSNP99.90to99.95",
            "variant_FILTER_VQSRTrancheSNP99.95to100.00+",
            "variant_FILTER_VQSRTrancheSNP99.95to100.00",
            "variant_FS",
            "variant_ID",
            "variant_InbreedingCoeff",
            "variant_LOF",
            "variant_MLEAC",
            "variant_MLEAF",
            "variant_MQ",
            "variant_MQRankSum",
            "variant_NEGATIVE_TRAIN_SITE",
            "variant_NMD",
            "variant_POSITIVE_TRAIN_SITE",
            "variant_QD",
            "variant_QUAL",
            "variant_RAW_MQandDP",
            "variant_ReadPosRankSum",
            "variant_RegionType",
            "variant_SOR",
            "variant_VQSLOD",
            #   "variant_altlen",
            "variant_culprit",
            "variant_set",
        }
        dimensions = {
            "alleles",
            "ploidy",
            "samples",
            "variants",
            "alt_alleles",
            "genotypes",
            "sb_statistics",
        }
        dim_variant_alt_allele_variable = [
            "variant_AC",
            "variant_AF",
            "variant_ANN_AA_length",
            "variant_ANN_AA_pos",
            "variant_ANN_Allele",
            "variant_ANN_Annotation",
            "variant_ANN_Annotation_Impact",
            "variant_ANN_CDS_length",
            "variant_ANN_CDS_pos",
            "variant_ANN_Distance",
            "variant_ANN_Feature_ID",
            "variant_ANN_Feature_Type",
            "variant_ANN_Gene_ID",
            "variant_ANN_Gene_Name",
            "variant_ANN_HGVS_c",
            "variant_ANN_HGVS_p",
            "variant_ANN_Rank",
            "variant_ANN_Transcript_BioType",
            "variant_ANN_cDNA_length",
            "variant_ANN_cDNA_pos",
            "variant_AS_BaseQRankSum",
            "variant_AS_FS",
            "variant_AS_InbreedingCoeff",
            "variant_AS_MQ",
            "variant_AS_MQRankSum",
            "variant_AS_QD",
            "variant_AS_ReadPosRankSum",
            "variant_AS_SOR",
            "variant_MLEAC",
            "variant_MLEAF",
            #   "variant_altlen",
        ]
    else:
        expected_data_vars = {
            "variant_allele",
            "variant_filter_pass",
            "variant_is_snp",
            "variant_numalt",
            "variant_CDS",
            "call_genotype",
            "call_AD",
        }
        dimensions = {"alleles", "ploidy", "samples", "variants"}
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "variant_position",
        "variant_chrom",
        "sample_id",
    }

    assert set(ds.coords) == expected_coords

    # check dimensions
    assert set(ds.dims) == dimensions

    # check dim lengths
    df_samples = pf8.sample_metadata()
    n_samples = len(df_samples)
    n_variants = ds.sizes["variants"]
    assert ds.sizes["samples"] == n_samples
    assert ds.sizes["ploidy"] == 2
    assert ds.sizes["alleles"] == 7

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xarray.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert x.ndim, f == 2
            assert x.shape == (n_variants, 7)
            assert x.dims == ("variants", "alleles")
        elif extended and f in dim_variant_alt_allele_variable:
            assert x.ndim, f == 1
            assert x.shape == (n_variants, 6)
            assert x.dims == ("variants", "alt_alleles")
        elif f == "variant_RAW_MQandDP":
            assert x.ndim, f == 1
            assert x.shape == (n_variants, 2)
            assert x.dims == ("variants", "ploidy")
        elif f.startswith("variant_"):
            assert x.ndim, f == 1
            assert x.shape == (n_variants,)
            assert x.dims == ("variants",)
        elif f == "call_genotype":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape == (n_variants, n_samples, 2)
        elif f == "call_AD":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "alleles")
            assert x.shape == (n_variants, n_samples, 7)
        elif f == "call_PL":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "genotypes")
            assert x.shape == (n_variants, n_samples, 3)
        elif f == "call_SB":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "sb_statistics")
            assert x.shape == (n_variants, n_samples, 4)
        elif f.startswith("call_"):
            assert x.ndim, f == 2
            assert x.dims == ("variants", "samples")
            assert x.shape == (n_variants, n_samples)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)
    # check variant_filter_pass
    filter_pass = ds["variant_filter_pass"].values
    n_pass = np.count_nonzero(filter_pass)
    assert n_pass < n_variants

    # check can setup computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xarray.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xarray.DataArray)


@pytest.mark.parametrize(
    "region",
    [
        "Pf3D7_01_v3",
        "*",
        ["Pf3D7_07_v3", "Pf3D7_02_v3", "Pf3D7_03_v3"],
        ["Pf3D7_07_v3", "Pf3D7_02_v3:15-20", "Pf3D7_03_v3:40-50"],
        "PF3D7_0920900.1",
    ],
)
def test_genome_sequence(region):
    pf8 = setup_pf8()

    seq = pf8.genome_sequence(region=region)
    assert isinstance(seq, da.Array)
    assert seq.dtype == "|S1"


@pytest.mark.parametrize(
    "attributes",
    [
        ("ID", "Parent", "Name"),
        "*",
        ["ID"],
    ],
)
def test_genome_features(attributes):
    pf8 = setup_pf8()

    default_columns = [
        "contig",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
    ]
    # check fields
    df = pf8.genome_features(attributes=attributes)
    assert isinstance(df, pd.DataFrame)
    if attributes == "*":
        additional_columns = [
            "ID",
            "Name",
            "Note",
            "Parent",
            "description",
            "gene_id",
            "protein_source_id",
        ]
        expected_columns = default_columns + additional_columns
    else:
        expected_columns = default_columns + list(attributes)
    assert list(df.columns) == expected_columns

    # check dimensions
    expected_len = 50070
    assert df.shape == (expected_len, len(expected_columns))
