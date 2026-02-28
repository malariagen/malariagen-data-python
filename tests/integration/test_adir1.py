import numpy as np
import pytest

from malariagen_data import Adir1, Region
from malariagen_data.util import _locate_region, _resolve_region


def setup_adir1(url="simplecache::gs://vo_adir_release_master_us_central1/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Af1 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Adir1(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Adir1(url, **kwargs)


def test_repr():
    adir1 = setup_adir1(check_location=True)
    assert isinstance(adir1, Adir1)
    r = repr(adir1)
    assert isinstance(r, str)


@pytest.mark.parametrize(
    "region_raw",
    [
        "ADIR015707",
        "KB672490",
        "KB672490:4871446-4871535",
        "KB672490:2,630,355-2,633,221",
        Region("KB672490", 4871446, 4871535),
    ],
)
def test_locate_region(region_raw):
    # Migrated to tests/anoph/test_base.py::test_locate_region_adir1
    adir1 = setup_adir1()
    gene_annotation = adir1.geneset(attributes=["ID"])
    region = _resolve_region(adir1, region_raw)
    pos = adir1.snp_sites(region=region.contig, field="POS")
    ref = adir1.snp_sites(region=region.contig, field="REF")
    loc_region = _locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "KB672490":
        assert region.contig == "KB672490"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "ADIR015707":
        gene = gene_annotation.query("ID == 'ADIR015707'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute() == np.array(["T", "T", "G", "T", "T"])
        ).all()

    # check string parsing
    if region_raw == "KB672490:4871446-4871535":
        assert region == Region("KB672490", 4871446, 4871535)
    if region_raw == "KB672490:2,630,355-2,633,221":
        assert region == Region("KB672490", 2630355, 2633221)
