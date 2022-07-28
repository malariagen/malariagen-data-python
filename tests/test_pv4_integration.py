import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray

from malariagen_data.pv4 import Pv4


def setup_pv4(url="simplecache::gs://pv4_staging/", **storage_kwargs):
    if url.startswith("simplecache::"):
        storage_kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Pv4(url, **storage_kwargs)


@pytest.mark.parametrize(
    "url",
    [
        "gs://pv4_staging/",
        "gcs://pv4_staging/",
        "gs://pv4_staging",
        "gcs://pv4_staging",
        "simplecache::gs://pv4_staging/",
        "simplecache::gcs://pv4_staging/",
    ],
)
def test_sample_metadata(url):

    pv4 = setup_pv4(url)
    df_samples = pv4.sample_metadata()

    expected_cols = (
        "Sample",
        "Study",
        "Site",
        "First-level administrative division",
        "Country",
        "Lat",
        "Long",
        "Year",
        "ENA",
        "All samples same individual",
        "Population",
        "% callable",
        "QC pass",
        "Exclusion reason",
        "Is returning traveller",
    )

    assert tuple(df_samples.columns) == expected_cols
    expected_len = 1895
    assert len(df_samples) == expected_len


@pytest.mark.parametrize("extended", [True, False])
def test_variant_calls(extended):

    pv4 = setup_pv4()

    ds = pv4.variant_calls(extended=extended)
    assert isinstance(ds, xarray.Dataset)

    # check fields
    if extended:
        expected_data_vars = {
            "variant_allele",
            "call_AD",
            "call_genotype",
            "variant_FILTER_Apicoplast",
            "variant_FS",
            "variant_MULTIALLELIC",
            "variant_SNPEFF_AMINO_ACID_CHANGE",
            "variant_SNPEFF_EFFECT",
            "variant_is_snp",
            "variant_SNPEFF_IMPACT",
            "variant_FILTER_Low_VQSLOD",
            "variant_VariantType",
            "variant_FILTER_Centromere",
            "variant_SNPEFF_EXON_ID",
            "variant_AF",
            "variant_FILTER_InternalHypervariable",
            "variant_FILTER_Mitochondrion",
            "variant_RegionType",
            "variant_SNPEFF_TRANSCRIPT_ID",
            "variant_SOR",
            "variant_FILTER_ShortContig",
            "variant_VQSLOD",
            "call_DP",
            "variant_filter_pass",
            "variant_ID",
            "variant_FILTER_SubtelomericHypervariable",
            "variant_QD",
            "variant_MQ",
            "call_PGT",
            "variant_CDS",
            "variant_QUAL",
            "variant_DP",
            "variant_altlen",
            "variant_AC",
            "call_PL",
            "variant_SNPEFF_GENE_NAME",
            "call_GQ",
            "variant_numalt",
            "variant_SNPEFF_CODON_CHANGE",
            "variant_AN",
            "variant_SNPEFF_FUNCTIONAL_CLASS",
            "call_PID",
        }

        dimensions = {
            "alleles",
            "ploidy",
            "samples",
            "variants",
            "alt_alleles",
            "genotypes",
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
            "variant_altlen",
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
    df_samples = pv4.sample_metadata()
    n_samples = len(df_samples)
    n_variants = ds.dims["variants"]
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 7

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
        "PvP01_05_v1",
        "*",
        ["PvP01_01_v1", "PvP01_05_v1", "PvP01_MIT_v1"],
        ["PvP01_01_v1", "PvP01_05_v1:15-20", "PvP01_MIT_v1:40-50"],
        "PVP01_0100100.1:pep",
    ],
)
def test_genome_sequence(region):

    pv4 = setup_pv4()

    seq = pv4.genome_sequence(region=region)
    assert isinstance(seq, da.Array)
    assert seq.dtype == "S1"


@pytest.mark.parametrize(
    "attributes",
    [
        ("ID", "Parent", "Name", "alias"),
        "*",
        ["ID", "literature"],
    ],
)
def test_genome_features(attributes):

    pv4 = setup_pv4()

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
    df = pv4.genome_features(attributes=attributes)
    assert isinstance(df, pd.DataFrame)
    if attributes == "*":
        additional_columns = [
            "Dbxref",
            "Derives_from",
            "End_range",
            "ID",
            "Name",
            "Note",
            "Ontology_term",
            "Parent",
            "Start_range",
            "alias",
            "comment",
            "cytoplasmic_polypeptide_region",
            "eupathdb_uc",
            "gPI_anchor_cleavage_site",
            "literature",
            "membrane_structure",
            "non_cytoplasmic_polypeptide_region",
            "orthologous_to",
            "polypeptide_domain",
            "previous_systematic_id",
            "product",
            "signal_peptide",
            "stop_codon_redefined_as_selenocysteine",
            "synonym",
            "translation",
            "transmembrane_polypeptide_region",
        ]
        expected_columns = default_columns + additional_columns
    else:
        expected_columns = default_columns + list(attributes)
    assert list(df.columns) == expected_columns

    # check dimensions
    expected_len = 38681
    assert df.shape == (expected_len, len(expected_columns))
