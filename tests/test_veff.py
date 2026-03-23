"""Tests for the Annotator genome cache in veff.py."""

import numpy as np
import pandas as pd

from malariagen_data.veff import Annotator


def _make_genome(contigs):
    """Create a minimal mock genome (dict-like) mapping contig names to
    numpy byte arrays."""
    genome = {}
    for name in contigs:
        seq = np.frombuffer(f"ATCGATCG{name}".encode(), dtype="S1")
        genome[name] = seq
    return genome


def _make_genome_features():
    """Return a minimal genome_features DataFrame with the required columns."""
    return pd.DataFrame(
        {
            "ID": ["gene1"],
            "Parent": ["root"],
            "type": ["gene"],
            "start": [1],
            "end": [100],
            "contig": ["chr1"],
            "strand": ["+"],
        }
    )


class TestGenomeCacheDefaultMaxsize:
    """Verify that the default cache maxsize is 5."""

    def test_default_maxsize(self):
        genome = _make_genome(["chr1"])
        ann = Annotator(genome=genome, genome_features=_make_genome_features())
        cache_info = ann._load_genome_seq.cache_info()
        assert cache_info.maxsize == 5


class TestGenomeCacheLRUEviction:
    """Verify that the LRU cache evicts the oldest entry when full."""

    def test_eviction(self):
        contigs = ["chr1", "chr2", "chr3"]
        genome = _make_genome(contigs)
        ann = Annotator(
            genome=genome,
            genome_features=_make_genome_features(),
            genome_cache_maxsize=2,
        )

        # Load all three contigs in order.
        for c in contigs:
            ann._load_genome_seq(c)

        info = ann._load_genome_seq.cache_info()
        # Only 2 entries should remain (chr2 and chr3).
        assert info.currsize == 2
        # 3 total misses (each first access is a miss).
        assert info.misses == 3

        # Accessing chr1 again should be a miss because it was evicted.
        ann._load_genome_seq("chr1")
        info = ann._load_genome_seq.cache_info()
        assert info.misses == 4


class TestClearGenomeCache:
    """Verify that clear_genome_cache() empties the cache."""

    def test_clear(self):
        genome = _make_genome(["chr1", "chr2"])
        ann = Annotator(genome=genome, genome_features=_make_genome_features())

        ann._load_genome_seq("chr1")
        ann._load_genome_seq("chr2")
        assert ann._load_genome_seq.cache_info().currsize == 2

        ann.clear_genome_cache()
        assert ann._load_genome_seq.cache_info().currsize == 0


class TestGenomeCacheUnbounded:
    """Verify that maxsize=None gives an unbounded cache."""

    def test_unbounded(self):
        contigs = [f"chr{i}" for i in range(20)]
        genome = _make_genome(contigs)
        ann = Annotator(
            genome=genome,
            genome_features=_make_genome_features(),
            genome_cache_maxsize=None,
        )

        for c in contigs:
            ann._load_genome_seq(c)

        info = ann._load_genome_seq.cache_info()
        assert info.maxsize is None
        assert info.currsize == 20


class TestPerInstanceCacheIsolation:
    """Verify that two Annotator instances have independent caches."""

    def test_isolation(self):
        genome = _make_genome(["chr1", "chr2"])
        features = _make_genome_features()

        ann1 = Annotator(genome=genome, genome_features=features)
        ann2 = Annotator(genome=genome, genome_features=features)

        ann1._load_genome_seq("chr1")
        assert ann1._load_genome_seq.cache_info().currsize == 1
        assert ann2._load_genome_seq.cache_info().currsize == 0

        ann2._load_genome_seq("chr2")
        assert ann1._load_genome_seq.cache_info().currsize == 1
        assert ann2._load_genome_seq.cache_info().currsize == 1

        ann1.clear_genome_cache()
        assert ann1._load_genome_seq.cache_info().currsize == 0
        assert ann2._load_genome_seq.cache_info().currsize == 1
