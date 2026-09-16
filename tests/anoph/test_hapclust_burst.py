import numpy as np
import pytest

from malariagen_data.anoph.hapclust_burst import (
    burst_clusters,
    estimate_background_diversity,
)


def simulate_haplotypes(rng, cluster_sizes, n_background=60, n_sites=400, within=0.005):
    """Background haplotypes plus tight clusters around random founders."""
    haps, truth = [], []
    for _ in range(n_background):
        haps.append(rng.random(n_sites) < 0.3)
        truth.append(0)
    for cluster_id, size in enumerate(cluster_sizes, start=1):
        founder = rng.random(n_sites) < 0.3
        for _ in range(size):
            haps.append(founder ^ (rng.random(n_sites) < within))
            truth.append(cluster_id)
    haps = np.array(haps, dtype=np.int8)
    dist = (haps[:, None, :] != haps[None, :, :]).sum(-1).astype(float)
    return dist, np.array(truth)


def agreement(labels, truth):
    """Fraction of pairs that the two labellings agree about."""
    same_a = labels[:, None] == labels[None, :]
    same_b = truth[:, None] == truth[None, :]
    iu = np.triu_indices(len(labels), 1)
    return float((same_a[iu] == same_b[iu]).mean())


def test_recovers_planted_clusters():
    rng = np.random.default_rng(1)
    dist, truth = simulate_haplotypes(rng, [30, 20])
    labels = burst_clusters(dist)
    assert labels.max() == 2
    assert agreement(labels, truth) > 0.95


def test_no_clusters_without_structure():
    rng = np.random.default_rng(2)
    dist, _ = simulate_haplotypes(rng, [], n_background=150)
    assert burst_clusters(dist).max() == 0


def test_labels_are_contiguous_and_size_ordered():
    rng = np.random.default_rng(3)
    dist, _ = simulate_haplotypes(rng, [40, 15])
    labels = burst_clusters(dist)
    ids = [i for i in np.unique(labels) if i > 0]
    assert ids == list(range(1, len(ids) + 1))
    sizes = [(labels == i).sum() for i in ids]
    assert sizes == sorted(sizes, reverse=True)


def test_invariant_to_sample_size():
    """The same structure at two sample sizes gives the same partition."""
    rng = np.random.default_rng(4)
    dist_small, truth_small = simulate_haplotypes(rng, [30, 20], n_background=60)
    rng = np.random.default_rng(4)
    dist_big, truth_big = simulate_haplotypes(rng, [150, 100], n_background=300)
    small = burst_clusters(dist_small)
    big = burst_clusters(dist_big)
    assert small.max() == big.max() == 2
    assert agreement(small, truth_small) > 0.95
    assert agreement(big, truth_big) > 0.95


def test_size_floor_applies():
    rng = np.random.default_rng(5)
    dist, _ = simulate_haplotypes(rng, [30, 6])
    assert burst_clusters(dist, min_cluster_size=5, min_cluster_freq=0).max() == 2
    assert burst_clusters(dist, min_cluster_size=10, min_cluster_freq=0).max() == 1


def test_background_can_be_supplied():
    """A background far above the truth makes ordinary clades look swept."""
    rng = np.random.default_rng(6)
    dist, _ = simulate_haplotypes(rng, [], n_background=120)
    estimated = estimate_background_diversity(dist)
    assert burst_clusters(dist, background=estimated).max() == 0
    assert burst_clusters(dist, background=estimated * 10).max() > 0


def test_tiny_input_returns_no_clusters():
    assert burst_clusters(np.zeros((3, 3))).tolist() == [0, 0, 0]


@pytest.mark.parametrize("linkage_method", ["average", "complete", "single"])
def test_works_with_other_linkage_methods(linkage_method):
    rng = np.random.default_rng(7)
    dist, truth = simulate_haplotypes(rng, [30, 20])
    labels = burst_clusters(dist, linkage_method=linkage_method)
    assert agreement(labels, truth) > 0.9
