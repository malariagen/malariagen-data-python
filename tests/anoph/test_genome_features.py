import bokeh.plotting
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.genome_features import AnophelesGenomeFeaturesData
from malariagen_data.util import Region, resolve_region

# Global RNG for test file; functions may override with local RNG for reproducibility
rng = np.random.default_rng(seed=42)


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesGenomeFeaturesData(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesGenomeFeaturesData(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


gff3_cols = [
    "contig",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
]


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_no_attributes(fixture, api: AnophelesGenomeFeaturesData):
    df_gf = api.genome_features(attributes=None)
    assert isinstance(df_gf, pd.DataFrame)
    expected_cols = gff3_cols + ["attributes"]
    assert df_gf.columns.to_list() == expected_cols
    assert len(df_gf) > 0
    for contig in df_gf["contig"].unique():
        assert contig in fixture.contigs


def test_genome_features_default_attributes_ag3(
    ag3_sim_api: AnophelesGenomeFeaturesData,
):
    df_gf = ag3_sim_api.genome_features()
    assert isinstance(df_gf, pd.DataFrame)
    expected_cols = gff3_cols + ["ID", "Parent", "Name", "description"]
    assert df_gf.columns.to_list() == expected_cols


def test_genome_features_default_attributes_af1(
    af1_sim_api: AnophelesGenomeFeaturesData,
):
    df_gf = af1_sim_api.genome_features()
    assert isinstance(df_gf, pd.DataFrame)
    expected_cols = gff3_cols + ["ID", "Parent", "Note", "description"]
    assert df_gf.columns.to_list() == expected_cols


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_region_contig(fixture, api: AnophelesGenomeFeaturesData):
    for contig in fixture.contigs:
        df_gf = api.genome_features(region=contig, attributes=None)
        expected_cols = gff3_cols + ["attributes"]
        assert df_gf.columns.to_list() == expected_cols
        assert len(df_gf) > 0
        assert (df_gf["contig"] == contig).all()


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_region_string(fixture, api: AnophelesGenomeFeaturesData):
    parametrize_region = [
        # Single contig.
        fixture.random_contig(),
        # List of contigs.
        [fixture.random_contig(), fixture.random_contig()],
        # Single region.
        fixture.random_region_str(),
        # List of regions.
        [fixture.random_region_str(), fixture.random_region_str()],
    ]

    for region in parametrize_region:
        df_gf = api.genome_features(region=region, attributes=None)
        expected_cols = gff3_cols + ["attributes"]
        assert df_gf.columns.to_list() == expected_cols
        # N.B., it's possible that the region overlaps no features.
        r = resolve_region(api, region)
        if len(df_gf) > 0 and isinstance(r, Region):
            assert (df_gf["contig"] == r.contig).all()
            if r.start is not None:
                assert (df_gf["end"] >= r.start).all()
            if r.end is not None:
                assert (df_gf["start"] <= r.end).all()


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_genes(fixture, api: AnophelesGenomeFeaturesData):
    for contig in fixture.contigs:
        fig = api.plot_genes(region=contig, show=False)
        assert isinstance(fig, bokeh.plotting.figure)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_genes_with_gene_labels(fixture, api: AnophelesGenomeFeaturesData):
    # For each contig in the fixture...
    for contig in fixture.contigs:
        # Get the genes for this contig.
        genes_df = api.genome_features(region=contig).query("type == 'gene'")

        # If there are no genes, we cannot label them.
        if not genes_df.empty:
            # Get a random number of genes to sample.
            random_genes_n = rng.integers(low=1, high=len(genes_df) + 1)

            # Get a random sample of genes.
            random_sample_genes_df = genes_df.sample(n=random_genes_n)

            # Put the random gene "ID" and its "Name" in a dictionary.
            random_gene_labels = dict(
                zip(random_sample_genes_df["ID"], random_sample_genes_df["Name"])
            )

            # Check that we get a Bokeh figure from plot_genes() with these gene_labels.
            fig = api.plot_genes(
                region=contig, show=False, gene_labels=random_gene_labels
            )
            assert isinstance(fig, bokeh.plotting.figure)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_transcript(fixture, api: AnophelesGenomeFeaturesData):
    for contig in fixture.contigs:
        df_transcripts = api.genome_features(region=contig).query("type == 'mRNA'")
        transcript = rng.choice(df_transcripts["ID"].values)
        fig = api.plot_transcript(transcript=transcript, show=False)
        assert isinstance(fig, bokeh.plotting.figure)


@pytest.fixture
def gh334_api(fixture_dir):
    return AnophelesGenomeFeaturesData(
        url=(fixture_dir / "gh334").as_uri(),
        public_url=(fixture_dir / "gh334").as_uri(),
        config_path="config.json",
        major_version_number=1,
        major_version_path="v1.0",
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent"),
    )


def test_gh334(gh334_api):
    # Accommodate exons with multiple parents
    # https://github.com/malariagen/malariagen-data-python/issues/334
    transcript = "LOC125767311_t2"
    df_gf = gh334_api.genome_feature_children(parent=transcript)
    assert len(df_gf) == 20


@pytest.mark.parametrize("chrom", ["2RL", "3RL"])
def test_genome_features_virtual_contigs(ag3_sim_api, chrom):
    api = ag3_sim_api
    contig_r, contig_l = api.virtual_contigs[chrom]
    df_r = api.genome_features(region=contig_r)
    df_l = api.genome_features(region=contig_l)
    max_r = api.genome_sequence(contig_r).shape[0]
    df_l = df_l.assign(start=lambda x: x.start + max_r, end=lambda x: x.end + max_r)
    df_concat = pd.concat([df_r, df_l], axis=0).reset_index(drop=True)
    df_concat = df_concat.assign(contig=chrom)
    df = api.genome_features(region=chrom)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == df_r.shape[0] + df_l.shape[0]
    assert all(df["contig"] == chrom)
    assert_frame_equal(df, df_concat)

    # Test with region.
    seq = api.genome_sequence(region=chrom)
    start, stop = sorted(rng.integers(low=1, high=len(seq), size=2))
    region = f"{chrom}:{start:,}-{stop:,}"
    df = api.genome_features(region=region)
    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert df["contig"].unique() == region.split(":")[0]
