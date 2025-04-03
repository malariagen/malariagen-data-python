import numpy as np
import pytest

from malariagen_data import Af1, Region
from malariagen_data.util import locate_region, resolve_region


def setup_af1(url="simplecache::gs://vo_afun_release_master_us_central1/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Af1 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Af1(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Af1(url, **kwargs)


def test_repr():
    af1 = setup_af1(check_location=True)
    assert isinstance(af1, Af1)
    r = repr(af1)
    assert isinstance(r, str)


@pytest.mark.parametrize(
    "region_raw",
    [
        "LOC125762289",
        "X",
        "2RL:48714463-48715355",
        "2RL:24,630,355-24,633,221",
        Region("2RL", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
    af1 = setup_af1()
    gene_annotation = af1.geneset(attributes=["ID"])
    region = resolve_region(af1, region_raw)
    pos = af1.snp_sites(region=region.contig, field="POS")
    ref = af1.snp_sites(region=region.contig, field="REF")
    loc_region = locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "X":
        assert region.contig == "X"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "LOC125762289":
        gene = gene_annotation.query("ID == 'LOC125762289'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute()
            == np.array(["T", "T", "T", "C", "T"], dtype="S1")
        ).all()

    # check string parsing
    if region_raw == "2RL:48714463-48715355":
        assert region == Region("2RL", 48714463, 48715355)
    if region_raw == "2RL:24,630,355-24,633,221":
        assert region == Region("2RL", 24630355, 24633221)


@pytest.mark.parametrize(
    "inversion",
    ["2La", "2Rb", "2Rc_col", "X_x"],
)
def test_karyotyping(inversion):
    af1 = setup_af1()

    with pytest.raises(NotImplementedError):
        af1.karyotype(
            inversion=inversion,
            sample_sets="1229-VO-GH-DADZIE-VMF00095",
            sample_query=None,
        )


@pytest.fixture(scope="module")
def af1():
    return Af1(
        "simplecache::gs://vo_afun_release_master_us_central1/",
        simplecache=dict(cache_storage="gcs_cache"),
        debug=True,
    )


def test_plot_haplotype_network_string_direct(af1, mocker):
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    af1.plot_haplotype_network(
        region="2RL:24,630,355-24,633,221",
        analysis="funestus",
        sample_sets="1.0",
        sample_query="taxon == 'funestus'",
        color="country",
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] == "partition"
    assert call_args["ht_color_counts"] is not None


def test_plot_haplotype_network_string_cohort(af1, mocker):
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    af1.plot_haplotype_network(
        region="2RL:24,630,355-24,633,221",
        analysis="funestus",
        sample_sets="1.0",
        sample_query="taxon == 'funestus'",
        color="year",
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] == "partition"
    assert call_args["ht_color_counts"] is not None


def test_plot_haplotype_network_mapping(af1, mocker):
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    color_mapping = {"2012": "year == 2012", "2014": "year == 2014"}
    af1.plot_haplotype_network(
        region="2RL:24,630,355-24,633,221",
        analysis="funestus",
        sample_sets="1.0",
        sample_query="taxon == 'funestus'",
        color=color_mapping,
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] == "partition"
    assert call_args["ht_color_counts"] is not None


def test_plot_haplotype_network_none(af1, mocker):
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    af1.plot_haplotype_network(
        region="2RL:24,630,355-24,633,221",
        analysis="funestus",
        sample_sets="1.0",
        sample_query="taxon == 'funestus'",
        color=None,
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] is None
    assert call_args["ht_color_counts"] is None
