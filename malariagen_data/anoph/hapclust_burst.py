"""Burst-based haplotype clustering.

Cutting a dendrogram at a fixed height confounds two things a sweep separates:
when a clade coalesced, and how fast it did. An old sweep is not shallow, and a
shallow clade is not necessarily a sweep, so no single height is right for a
whole dendrogram, and the height that works at one sample size does not work at
another.

This module instead scores every clade against the diversity of the window it
sits in, and selects the best set of non-overlapping clades. Two competing
models for the distance between a pair of haplotypes:

* background: ``d ~ Geometric(mean=bg)``, where ``bg`` is the neutral expected
  pairwise distance for the window;
* cluster c: ``d ~ Geometric(mean=theta_c)``, one fitted diversity per cluster,
  constrained to ``theta_c <= max_diversity * bg`` so that "cluster" means
  "much less diverse than the background" rather than merely "a group".

A clade scores the summed log-likelihood ratio over the pairs inside it. Since
the reference model is "every pair is background", pairs outside every cluster
contribute exactly zero, so the objective is separable over non-overlapping
clades and the best set follows from one bottom-up pass (the excess-of-mass
argument used by HDBSCAN, with a diversity weight in place of a density).

Scores scale with the number of pairs in a clade, so scaling every clade's size
leaves the selected set unchanged: the clustering behaves the same at 100 and at
2,000 haplotypes. Calling thresholds are therefore per pair (an effect size)
plus a size floor, never a total.

The geometric family is used because pairwise differences under the neutral
coalescent are geometric, and because it makes the merge test below depend only
on ratios of distances.
"""

from typing import Optional, Tuple

import numpy as np


def _geometric_loglik(n_pairs: float, sum_d: float, mean: float) -> float:
    """Total log-likelihood of `n_pairs` distances summing to `sum_d`."""
    m = max(mean, 1e-9)
    return sum_d * np.log(m / (1.0 + m)) - n_pairs * np.log1p(m)


def _geometric_fit(m: np.ndarray) -> np.ndarray:
    """Per-pair maximised geometric log-likelihood at mean `m`."""
    m = np.maximum(m, 1e-12)
    return m * np.log(m) - (1.0 + m) * np.log1p(m)


def _median_pair_distance(
    dist: np.ndarray, rows: Optional[np.ndarray] = None, block: int = 512
) -> float:
    """Median distance over distinct pairs, without materialising the pairs.

    Distances are counts, so a histogram gives the median exactly, and
    accumulating it a block of rows at a time keeps memory proportional to the
    largest distance rather than to the number of pairs. The obvious
    ``dist[np.triu_indices(n, 1)]`` allocates two int64 indices per pair, which
    reaches several gigabytes at the sample sizes this is meant to run at.

    `rows` restricts the calculation to a subset of haplotypes, again without
    taking a copy of the submatrix.
    """
    all_rows = rows is None
    idx = np.arange(dist.shape[0]) if all_rows else np.asarray(rows)
    m = idx.size
    if m < 2:
        return 0.0

    # The matrix is symmetric, so count every off-diagonal entry and halve.
    counts = np.zeros(1, dtype=np.int64)
    for a in range(0, m, block):
        # Whole rows are a view; a subset needs a copy of the block.
        sub = dist[a : a + block] if all_rows else dist[np.ix_(idx[a : a + block], idx)]
        c = np.bincount(np.maximum(np.rint(sub).astype(np.int64).ravel(), 0))
        if c.size > counts.size:
            c[: counts.size] += counts
            counts = c
        else:
            counts[: c.size] += c
    counts[0] -= m  # the diagonal
    counts //= 2

    total = int(counts.sum())
    if total == 0:
        return 0.0
    cum = np.cumsum(counts)
    lo = int(np.searchsorted(cum, (total + 1) // 2))
    hi = int(np.searchsorted(cum, total // 2 + 1))
    return (lo + hi) / 2


def estimate_background_diversity(
    dist: np.ndarray, collapse_frac: float = 0.05
) -> float:
    """Estimate the neutral expected pairwise distance for a window.

    A sweep contributes many near-identical haplotypes, which drags the mean
    pairwise distance down. Collapsing near-identical haplotypes to a single
    representative removes that multiplicity, so the median distance among
    representatives approximates the neutral level.

    This is only reliable when unswept haplotypes are present. Where a region is
    swept end to end the estimate is far too low, and a value should be supplied
    explicitly instead, e.g. the mean pairwise distance seen in a comparable
    region without a sweep.
    """
    n = dist.shape[0]
    tol = collapse_frac * _median_pair_distance(dist)
    keep: list = []
    for i in range(n):
        if not keep or dist[i, keep].min() > tol:
            keep.append(i)
    if len(keep) < 4:
        keep = list(range(n))
    return float(_median_pair_distance(dist, np.asarray(keep)))


def _leaf_order(
    children: np.ndarray, size: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Lay tips out so that every node owns a contiguous block.

    Storing a start offset per node rather than an explicit list of tips keeps
    memory linear in the number of haplotypes.
    """
    start = np.zeros(2 * n - 1, dtype=np.int64)
    for v in range(2 * n - 2, n - 1, -1):
        a, b = children[v - n]
        start[a] = start[v]
        start[b] = start[v] + size[a]
    order = np.empty(n, dtype=np.int64)
    order[start[:n]] = np.arange(n)
    return order, start


def burst_clusters(
    dist: np.ndarray,
    background: Optional[float] = None,
    linkage_method: str = "average",
    min_cluster_size: int = 5,
    min_cluster_freq: float = 0.02,
    min_effect: float = 0.25,
    max_diversity: float = 0.5,
    max_split_loss: float = 0.15,
    resolution: float = 1.0,
) -> np.ndarray:
    """Assign haplotypes to swept clusters.

    Parameters
    ----------
    dist
        Square matrix of pairwise distances between haplotypes, as counts of
        differing SNPs.
    background
        Neutral expected pairwise distance for the window. Estimated from the
        data if not given, which assumes unswept haplotypes are present.
    linkage_method
        Linkage for the candidate tree. Average linkage is the default because
        its node heights are mean distances, which do not drift with sample size.
    min_cluster_size, min_cluster_freq
        Size floor for a called cluster: ``max(min_cluster_size,
        min_cluster_freq * n)``.
    min_effect
        Minimum log-likelihood ratio per pair for a clade to be called, in nats.
    max_diversity
        Upper bound on a cluster's diversity as a fraction of `background`.
    max_split_loss
        Two daughter clades are kept as one cluster while merging them costs at
        most this much per pair. Because the loss depends only on the ratio of
        the between-daughter distance to the within-daughter distance, this asks
        whether the daughters separated recently relative to their own age,
        rather than whether the split is deeper than some height.
    resolution
        Distance added to every group mean in the merge test, in SNPs. Without
        it, two clusters of identical haplotypes one mutation apart would have
        an infinite ratio.

    Returns
    -------
    Cluster assignment per haplotype: 0 for unassigned, then 1, 2, ... for
    clusters in decreasing size order.
    """
    from scipy.cluster.hierarchy import linkage as _linkage
    from scipy.spatial.distance import squareform

    dist = np.rint(np.asarray(dist, dtype=float))
    n = dist.shape[0]
    if n < 4:
        return np.zeros(n, dtype=np.int64)
    if background is None:
        background = estimate_background_diversity(dist)
    background = max(float(background), 1e-6)

    z = _linkage(squareform(dist, checks=False), method=linkage_method)
    children = z[:, :2].astype(np.int64)
    size = np.ones(2 * n - 1, dtype=np.int64)
    for i in range(n - 1):
        size[n + i] = size[children[i, 0]] + size[children[i, 1]]
    order, start = _leaf_order(children, size, n)

    # Work in leaf order so each node's pairs form a contiguous block.
    d = dist[np.ix_(order, order)]
    n_pairs = np.zeros(2 * n - 1)
    sum_d = np.zeros(2 * n - 1)
    bg_ll = np.zeros(2 * n - 1)
    log_bg_ratio = np.log(background / (1.0 + background))
    log1p_bg = np.log1p(background)

    for i in range(n - 1):
        v = n + i
        a, b = children[i]
        sa, sb = (
            slice(start[a], start[a] + size[a]),
            slice(start[b], start[b] + size[b]),
        )
        cross = d[sa, sb]
        n_pairs[v] = n_pairs[a] + n_pairs[b] + cross.size
        sum_d[v] = sum_d[a] + sum_d[b] + cross.sum()
        bg_ll[v] = (
            bg_ll[a] + bg_ll[b] + (cross.sum() * log_bg_ratio - cross.size * log1p_bg)
        )

    mean_d = np.divide(sum_d, n_pairs, out=np.zeros(2 * n - 1), where=n_pairs > 0)
    score = np.zeros(2 * n - 1)
    split_loss = np.zeros(2 * n - 1)
    for i in range(n - 1):
        v = n + i
        a, b = children[i]
        theta = min(mean_d[v], max_diversity * background)
        score[v] = _geometric_loglik(n_pairs[v], sum_d[v], theta) - bg_ll[v]
        # One diversity for the whole clade, or one per daughter plus one for
        # the pairs that cross between them?
        n_ab = n_pairs[v] - n_pairs[a] - n_pairs[b]
        s_ab = sum_d[v] - sum_d[a] - sum_d[b]
        groups = [(n_pairs[a], sum_d[a]), (n_pairs[b], sum_d[b]), (n_ab, s_ab)]
        split = sum(
            g_n * _geometric_fit(g_s / g_n + resolution)
            for g_n, g_s in groups
            if g_n > 0
        )
        merged = n_pairs[v] * _geometric_fit(sum_d[v] / n_pairs[v] + resolution)
        split_loss[v] = (split - merged) / n_pairs[v]

    # A clade may only be kept whole if no split inside it is too costly.
    cohesive = np.ones(2 * n - 1, dtype=bool)
    for i in range(n - 1):
        v = n + i
        a, b = children[i]
        cohesive[v] = split_loss[v] <= max_split_loss and cohesive[a] and cohesive[b]

    # Best set of non-overlapping clades, bottom up.
    best = np.zeros(2 * n - 1)
    keep_whole = np.zeros(2 * n - 1, dtype=bool)
    for i in range(n - 1):
        v = n + i
        a, b = children[i]
        own = score[v] if cohesive[v] else -np.inf
        below = best[a] + best[b]
        if own >= below:
            best[v], keep_whole[v] = own, True
        else:
            best[v] = below

    selected = []
    stack = [2 * n - 2]
    while stack:
        v = stack.pop()
        if v < n:
            continue
        if keep_whole[v]:
            selected.append(v)
        else:
            a, b = children[v - n]
            stack.extend([int(a), int(b)])

    size_floor = max(min_cluster_size, int(np.ceil(min_cluster_freq * n)))
    labels = np.zeros(n, dtype=np.int64)
    cid = 0
    for v in sorted(selected, key=lambda v: -size[v]):
        effect = score[v] / max(n_pairs[v], 1)
        if size[v] < size_floor or effect < min_effect:
            continue
        cid += 1
        labels[order[start[v] : start[v] + size[v]]] = cid
    return labels
