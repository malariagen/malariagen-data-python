import pytest

from malariagen_data import Adar1, Region
from malariagen_data.util import _locate_region, _resolve_region


def setup_adar1(url="simplecache::gs://vo_adar_release_master_us_central1/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Af1 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Adar1(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Adar1(url, **kwargs)


def test_repr():
    adar1 = setup_adar1(check_location=True)
    assert isinstance(adar1, Adar1)
    r = repr(adar1)
    assert isinstance(r, str)


@pytest.mark.parametrize(
    "region_raw",
    [
        "2",
        "gene-LOC125950257",
        "2:4871446-4871535",
        "2:2,630,355-2,633,221",
        Region("2", 4871446, 4871535),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
    adar1 = setup_adar1()
    gene_annotation = adar1.geneset(attributes=["ID"])
    region = _resolve_region(adar1, region_raw)
    pos = adar1.snp_sites(region=region.contig, field="POS")
    # Used by some code that has not been added yet
    # ref = adar1.snp_sites(region=region.contig, field="REF")
    loc_region = _locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "2":
        assert region.contig == "2"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "gene-LOC125950257":
        gene = gene_annotation.query("ID == 'gene-LOC125950257'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        # To be checked
        # assert (
        #    ref[loc_region][:5].compute() == np.array(["T", "T", "G", "T", "T"])
        # ).all()

    # check string parsing
    if region_raw == "2:4871446-4871535":
        assert region == Region("2", 4871446, 4871535)
    if region_raw == "2:2,630,355-2,633,221":
        assert region == Region("2", 2630355, 2633221)
