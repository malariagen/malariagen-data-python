import numpy as np
import pytest

from malariagen_data import Amin1, Region
from malariagen_data.util import locate_region, resolve_region


def setup_amin1(url="simplecache::gs://vo_adir_release_master_us_central1/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Af1 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Amin1(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Amin1(url, **kwargs)


def test_repr():
    amin1 = setup_amin1(check_location=True)
    assert isinstance(amin1, amin1)
    r = repr(amin1)
    assert isinstance(r, str)


@pytest.mark.parametrize(
    "region_raw",
    [
        "AMIN007480",
        "KB663655",
        "KB663655:3548753-3550841",
        "KB663655:3,548,753-3,550,841",
        Region("KB663655", 3548753, 3548753),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
    amin1 = setup_amin1()
    gene_annotation = amin1.geneset(attributes=["ID"])
    region = resolve_region(amin1, region_raw)
    pos = amin1.snp_sites(region=region.contig, field="POS")
    ref = amin1.snp_sites(region=region.contig, field="REF")
    loc_region = locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "KB663655":
        assert region.contig == "KB663655"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "AMIN007480":
        gene = gene_annotation.query("ID == 'AMIN007480'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute() == np.array(["T", "C", "A", "G", "A"])
        ).all()

    # check string parsing
    if region_raw == "KB663655:3548753-3550841":
        assert region == Region("KB672490", 3548753, 3550841)
    if region_raw == "KB663655:3,548,753-3,550,841":
        assert region == Region("KB663655", 3548753, 3550841)
