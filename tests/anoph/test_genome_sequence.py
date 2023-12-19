from typing import Mapping, Sequence

import dask.array as da
import numpy as np
import pytest
import zarr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.genome_sequence import AnophelesGenomeSequenceData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesGenomeSequenceData(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesGenomeSequenceData(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_contigs(fixture, api):
    contigs = api.contigs
    assert isinstance(contigs, tuple)
    assert all([isinstance(c, str) for c in contigs])
    assert contigs == tuple(fixture.contigs)


@parametrize_with_cases("fixture,api", cases=".")
def test_open_genome(fixture, api):
    root = api.open_genome()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in fixture.contigs:
        z = root[contig]
        assert isinstance(z, zarr.core.Array)
        assert z.ndim == 1
        assert z.dtype == "S1"


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_sequence(fixture, api):
    root = api.open_genome()
    for contig in fixture.contigs:
        seq = api.genome_sequence(region=contig)
        assert isinstance(seq, da.Array)
        assert seq.ndim == 1
        assert seq.dtype == "S1"
        assert seq.shape[0] == root[contig].shape[0]


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_sequence_region(fixture, api):
    for contig in fixture.contigs:
        contig_seq = api.genome_sequence(region=contig)
        # Pick a random start and stop position.
        start, stop = sorted(np.random.randint(low=1, high=len(contig_seq), size=2))
        region = f"{contig}:{start:,}-{stop:,}"
        seq = api.genome_sequence(region=region)
        assert isinstance(seq, da.Array)
        assert seq.ndim == 1
        assert seq.dtype.kind == "S"
        assert seq.shape[0] == stop - start + 1


def test_virtual_contigs(ag3_sim_api):
    api = ag3_sim_api
    contigs = api.contigs
    virtual_contigs = api.virtual_contigs
    assert isinstance(virtual_contigs, Mapping)
    for c, v in virtual_contigs.items():
        assert isinstance(c, str)
        assert isinstance(v, Sequence)
        for x in v:
            assert x in contigs


def test_genome_sequence_virtual_contigs(ag3_sim_api):
    api = ag3_sim_api
    for vc, contigs in api.virtual_contigs.items():
        contig_r, contig_l = contigs
        seq_r = api.genome_sequence(region=contig_r)
        seq_l = api.genome_sequence(region=contig_l)
        seq = api.genome_sequence(region=vc)
        assert isinstance(seq, da.Array)
        assert seq.dtype == seq_r.dtype == seq_l.dtype
        assert seq.shape[0] == seq_r.shape[0] + seq_l.shape[0]
        # N.B., we use a single-threaded computation here to avoid race conditions
        # when data are being cached locally from GCS (which manifests as blosc
        # decompression errors).
        assert da.all(seq == da.concatenate([seq_r, seq_l])).compute(
            scheduler="single-threaded"
        )

        # Test with region.
        start, stop = sorted(np.random.randint(low=1, high=len(seq), size=2))
        region = f"{vc}:{start:,}-{stop:,}"
        seq_region = api.genome_sequence(region=region)
        assert isinstance(seq_region, da.Array)
        assert seq_region.ndim == 1
        assert seq_region.dtype == seq.dtype
        assert seq_region.shape[0] == stop - start + 1
